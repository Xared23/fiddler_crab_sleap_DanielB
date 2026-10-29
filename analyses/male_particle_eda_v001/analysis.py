"""Finite, read-only male detection/experimental-tracklet EDA; run only in Slurm."""
from pathlib import Path
import os, json, hashlib, time, platform, sys
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, cdist
from scipy.signal import detrend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/v007_best_body_first_dataframe_v001'
BASE = Path(__file__).resolve().parent
DT = 1001 / 30000
W, H, NF = 7680, 4320, 28800
VIDEOS = ['Crab_Cam_2_2', 'Crab_Cam_2_3']
LAGS = np.array([1, 3, 10, 30, 60, 150, 300, 600, 1500])
RBINS = np.r_[0., np.geomspace(.1, 20000, 100)]
PBINS = np.arange(0, 2025, 25.)
SEED = 20261015
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 130})


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, data):
    def cast(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x))
    path.write_text(json.dumps(data, indent=2, default=cast, allow_nan=False) + '\n')


def segments(ids, frame):
    starts = np.r_[True, (ids[1:] != ids[:-1]) | (np.diff(frame) != 1)]
    seg = np.cumsum(starts) - 1
    return seg, np.bincount(seg)


def pairs(seg, frame, lag, stride=10):
    i = np.flatnonzero(frame % stride == 0)
    i = i[i + lag < len(seg)]
    return i[seg[i] == seg[i + lag]]


def displacement_stats(delta):
    """Uncentered radial moments; drift intentionally retained in MSD."""
    delta = np.asarray(delta, float)
    assert delta.ndim == 2 and delta.shape[1] == 2 and len(delta)
    assert np.isfinite(delta).all()
    r2 = np.sum(delta**2, axis=1)
    msd = float(r2.mean())
    return {'msd': msd, 'alpha2': float(np.mean(r2**2)/(2*msd**2)-1) if msd else np.nan,
            'overlap': float(np.mean(r2 < 100)), 'r2': r2}


def radial_histogram(r):
    hist = np.histogram(r, RBINS)[0]
    assert hist.sum() == len(r), 'Van Hove range insufficient'
    return hist, hist/len(r)/np.diff(RBINS)


def rectangle_pair_histogram(q, bins=PBINS):
    """Unordered pairs with translation correction for the full rectangular window."""
    n = len(q)
    ii,jj = np.triu_indices(n,1)
    delta = np.abs(q[ii]-q[jj]); r = np.linalg.norm(delta,axis=1)
    area = (W-delta[:,0])*(H-delta[:,1])
    assert np.all(area > 0)
    return np.histogram(r,bins,weights=W*H/area)[0]


def acf(x, maxlag=120):
    x = np.asarray(x, float)
    if x.ndim == 1: x = x[:, None]
    x = x - x.mean(axis=0)
    var = np.mean(np.sum(x*x, axis=1))
    if var == 0: return np.full(maxlag+1, np.nan)
    return np.array([np.mean(np.sum(x[:len(x)-k] * x[k:], axis=1))/var
                     for k in range(maxlag+1)])


def bootstrap_mean(values, frames, rng):
    block = np.minimum((frames * DT / 60).astype(int), 16)
    n = np.bincount(block, minlength=17)
    s = np.bincount(block, weights=values, minlength=17)
    draws = rng.integers(0, 17, (300, 17))
    den = n[draws].sum(axis=1)
    est = s[draws].sum(axis=1) / np.maximum(den, 1)
    return np.quantile(est[den > 0], [.025, .975])


def savefig(out, name, fig):
    fig.savefig(out/'figures'/f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(out/'figures'/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def tests():
    # No gaps, no cross-ID pairs; land removal must split a trajectory.
    ids = np.array([1,1,1,1,2,2]); f = np.array([0,1,3,4,0,1])
    s, lengths = segments(ids, f)
    assert lengths.tolist() == [2,2,2]
    assert pairs(s, f, 1, 1).tolist() == [0,2,4]
    assert len(pairs(s, f, 2, 1)) == 0
    f = np.arange(10); s = np.zeros(10, int)
    p = np.column_stack([2*f, np.zeros(10)])
    i = pairs(s, f, 3, 1)
    assert np.all(np.sum((p[i+3]-p[i])**2, axis=1) == 36)
    assert np.isclose(acf(np.arange(10), 0)[0], 1)
    # Translation correction: uniform rectangle should give g(r) ~ 1.
    rng = np.random.default_rng(SEED)
    xy = rng.random((1600, 2)) * [W, H]
    delta = xy[:, None] - xy[None, :]
    iu = np.triu_indices(len(xy), 1)
    dxy = delta[iu]; r = np.linalg.norm(dxy, axis=1)
    weights = W*H/((W-np.abs(dxy[:,0]))*(H-np.abs(dxy[:,1])))
    bins = np.arange(200, 2200, 200)
    g = np.histogram(r, bins, weights=weights)[0] / (
        len(xy)*(len(xy)-1)/2 * np.pi*np.diff(bins**2)/(W*H))
    assert np.max(np.abs(g-1)) < .15, g
    # Radial Gaussian density normalization and 2D non-Gaussian parameter.
    dr = rng.normal(size=(100000, 2)); r2 = np.sum(dr**2, axis=1)
    assert abs(np.mean(r2**2)/(2*np.mean(r2)**2)-1) < .02
    hist = np.histogram(np.sqrt(r2), RBINS)[0]/len(r2)/np.diff(RBINS)
    assert np.isclose(np.sum(hist*np.diff(RBINS)), 1)
    return {'passed': True, 'checks': ['gap and ID exclusion', 'ballistic MSD',
            'ACF normalization', 'uniform rectangle pair correction',
            'radial PDF normalization', 'Gaussian alpha2']}


def motion(df, video, population, out, rng):
    d = df.loc[df.accepted_for_tracking].copy()
    if population == 'land': d = d.loc[d.water.eq(False)].copy()
    d = d.sort_values(['tracklet_local_id', 'frame'])
    ids = d.tracklet_local_id.to_numpy(dtype=np.int64)
    frame = d.frame.to_numpy(dtype=np.int32)
    pos = d[['body_x','body_y']].to_numpy(float)
    scores = d.body_score.to_numpy(float)
    if population == 'all':
        same = ids[1:] == ids[:-1]
        assert np.all(np.diff(frame)[same] == 1), 'Unexpected original track gap'
    seg, lengths = segments(ids, frame)
    origins = np.flatnonzero(frame % 10 == 0)
    fixed = pairs(seg, frame, 300)
    rows, vanhove, vacfrows = [], [], []
    for lag in LAGS:
        i = pairs(seg, frame, int(lag))
        delta = pos[i+lag]-pos[i]
        stats = displacement_stats(delta)
        r2 = stats['r2']; r = np.sqrt(r2)
        msd = stats['msd']; lo, hi = bootstrap_mean(r2, frame[i], rng)
        sf = fixed if lag <= 300 else np.array([], int)
        # Endpoint score check is diagnostic, not a full-path confidence filter.
        high = (scores[i] >= .8) & (scores[i+lag] >= .8)
        row = dict(video=video, population=population, lag_frames=int(lag),
                   lag_s=lag*DT, n_pairs=len(i), n_segments=len(np.unique(seg[i])),
                   origin_survival=len(i)/len(origins), msd_px2=msd,
                   msd_block_low=lo, msd_block_high=hi,
                   rms_px=np.sqrt(msd), median_r_px=np.median(r),
                   p90_r_px=np.quantile(r,.9), p99_r_px=np.quantile(r,.99),
                   alpha2=stats['alpha2'],
                   overlap_10px=stats['overlap'],
                   mean_dx_px=delta[:,0].mean(), mean_dy_px=delta[:,1].mean(),
                   high_score_msd_px2=r2[high].mean() if high.any() else np.nan,
                   high_score_pairs=int(high.sum()),
                   fixed_origin_msd_px2=np.mean(np.sum((pos[sf+lag]-pos[sf])**2,axis=1)) if len(sf) else np.nan,
                   fixed_origin_pairs=len(sf))
        rows.append(row)
        hist, radial_pdf = radial_histogram(r)
        for k, n in enumerate(hist):
            vanhove.append(dict(video=video, population=population, lag_frames=int(lag),
                r_lo=RBINS[k], r_hi=RBINS[k+1], count=int(n), n_pairs=len(r),
                radial_pdf=radial_pdf[k]))
    # Coarse-grained velocity dt=10 frames; lag >=10 avoids interval overlap.
    for lag in [0,10,30,60,150,300]:
        i = pairs(seg, frame, lag+10)
        v0 = (pos[i+10]-pos[i])/(10*DT)
        vt = (pos[i+lag+10]-pos[i+lag])/(10*DT)
        dot = np.sum(v0*vt, axis=1)
        denom = np.sqrt(np.mean(np.sum(v0*v0,axis=1))*np.mean(np.sum(vt*vt,axis=1)))
        vc0 = v0-v0.mean(axis=0); vct = vt-vt.mean(axis=0)
        cden = np.sqrt(np.mean(np.sum(vc0*vc0,axis=1))*np.mean(np.sum(vct*vct,axis=1)))
        vacfrows.append(dict(video=video,population=population,lag_s=lag*DT,
                            n_pairs=len(i), vacf=dot.mean()/denom,
                            connected_vacf=np.mean(np.sum(vc0*vct,axis=1))/cden))
    pd.DataFrame(rows).to_csv(out/'tables'/f'{video}_{population}_motion.csv',index=False)
    pd.DataFrame(vanhove).to_csv(out/'tables'/f'{video}_{population}_vanhove.csv',index=False)
    pd.DataFrame(vacfrows).to_csv(out/'tables'/f'{video}_{population}_vacf.csv',index=False)
    # One-frame steps quantify the selection ceiling, not a physical speed limit.
    i = pairs(seg,frame,1,1); step = np.linalg.norm(pos[i+1]-pos[i],axis=1)
    assert np.max(step) <= 12.0001, 'Tracker step gate violated'
    summary = dict(n_observations=len(d), n_segments=len(lengths),
                   median_segment_frames=float(np.median(lengths)),
                   max_segment_frames=int(lengths.max()),
                   singleton_fraction=float(np.mean(lengths==1)),
                   one_frame_step_quantiles=np.quantile(step,[.5,.9,.99,1]),
                   fraction_steps_above_10px=float(np.mean(step>10)),
                   sample_origins=len(origins), fixed_origins=len(fixed))
    # Representative (not random) longest tracks: explicit selection, original pixels.
    if population == 'land':
        fig, ax = plt.subplots(figsize=(10,5.6))
        for ss in np.argsort(lengths)[-20:]:
            q = pos[seg == ss]
            ax.plot(q[::10,0],q[::10,1],lw=.8,alpha=.8)
            ax.scatter(q[0,0],q[0,1],s=8)
        ax.set(xlim=(0,W),ylim=(H,0),xlabel='x (image px)',ylabel='y (image px)',
               title=f'{video}: 20 longest continuous land segments (selection-biased)')
        ax.set_aspect('equal'); savefig(out,f'{video}_trajectories',fig)
    return pd.DataFrame(rows), pd.DataFrame(vanhove), pd.DataFrame(vacfrows), summary


def spatial(df, video, out):
    # Detection-only statistics do not require linking a physical identity.
    s = df.loc[df.accepted_for_tracking & (df.frame % 30 == 0)].copy()
    frames = np.arange(0,NF,30)
    clouds = {}; grids = {}; summaries = {}; pairframes = []; density_speed = []
    for pop in ['all','land']:
        d = s if pop == 'all' else s.loc[s.water.eq(False)]
        grouped = {int(f): z[['body_x','body_y']].to_numpy(float) for f,z in d.groupby('frame')}
        clouds[pop] = [grouped.get(int(f),np.empty((0,2))) for f in frames]
        grids[pop] = np.array([np.histogram2d(q[:,1],q[:,0],bins=(14,24),range=((0,H),(0,W)))[0]
                               for q in clouds[pop]])
        same = np.zeros(len(PBINS)-1); mixed = same.copy(); corrected = same.copy()
        ntotal = mtotal = 0; nn = []
        for t,q in enumerate(clouds[pop]):
            n = len(q)
            if n>1:
                dist = pdist(q)
                hist = np.histogram(dist,PBINS)[0]
                same += hist; ntotal += n*(n-1)/2
                nn.extend(cKDTree(q).query(q,k=2)[0][:,1])
                if pop=='all':
                    corrected += rectangle_pair_histogram(q)
                pairframes.append(dict(video=video,population=pop,frame=int(frames[t]),
                    count=n, pairs_under_100=int(np.sum(dist<100)),
                    median_nn=float(np.median(cKDTree(q).query(q,k=2)[0][:,1]))))
            # Circular +/- ~60, 120, 240 s offsets: same footprint, not independent animals.
            for shift in [60,120,240]:
                other = clouds[pop][(t+shift)%len(frames)]
                if n and len(other):
                    mixed += np.histogram(cdist(q,other),PBINS)[0]
                    mtotal += n*len(other)
        nn = np.asarray(nn)
        ratio = (same/ntotal)/np.maximum(mixed/mtotal,1e-30)
        g = corrected/(ntotal*np.pi*np.diff(PBINS**2)/(W*H)) if pop=='all' else np.full_like(same,np.nan)
        table = pd.DataFrame(dict(r_lo=PBINS[:-1],r_hi=PBINS[1:],same_pairs=same,
                         mixed_pairs=mixed,same_total_pairs=ntotal,mixed_total_pairs=mtotal,
                         same_to_time_mixed=ratio,g_rectangle=g))
        table.to_csv(out/'tables'/f'{video}_{pop}_pairs.csv',index=False)
        nnedges = np.arange(0,2005,5)
        nnhist = np.histogram(nn,nnedges)[0]
        pd.DataFrame(dict(r_lo=nnedges[:-1],r_hi=nnedges[1:],count=nnhist,
                         cdf=np.cumsum(nnhist)/len(nn))).to_csv(out/'tables'/f'{video}_{pop}_nn.csv',index=False)
        summaries[pop] = dict(sampled_frames=len(frames),sampled_detections=len(d),
            nn_quantiles_px=np.quantile(nn,[.1,.5,.9]),
            fraction_nn_under_100=float(np.mean(nn<100)),
            mixed_ratio_50_100=float((same[2:4].sum()/ntotal)/(mixed[2:4].sum()/mtotal)),
            mixed_ratio_100_200=float((same[4:8].sum()/ntotal)/(mixed[4:8].sum()/mtotal)))
    pd.DataFrame(pairframes).to_csv(out/'tables'/f'{video}_frame_spatial.csv',index=False)
    # Coarse density correlations: per-cell mean removed, not stationary assumed.
    arows = []
    for pop,grid in grids.items():
        x = grid.reshape(len(grid),-1)
        count = x.sum(axis=1)
        ca = acf(count); cd = acf(detrend(count)); cell = acf(x)
        # Remove per-frame total count and static shape together.
        residual = x-count[:,None]*(x.mean(axis=0)/count.mean())[None,:]
        shape = acf(residual)
        for lag in range(121):
            arows.append(dict(population=pop,lag_s=lag*30*DT,count_acf=ca[lag],
                             detrended_count_acf=cd[lag],cell_acf=cell[lag],shape_acf=shape[lag]))
        summaries[pop]['sample_count_fano'] = float(np.var(count)/np.mean(count))
    pd.DataFrame(arows).to_csv(out/'tables'/f'{video}_density_acf.csv',index=False)
    # 1-s displacements conditioned on nearest accepted land neighbor at origin.
    land = df.loc[df.accepted_for_tracking & df.water.eq(False)].sort_values(['tracklet_local_id','frame'])
    f = land.frame.to_numpy(int); p = land[['body_x','body_y']].to_numpy(float)
    sg,_ = segments(land.tracklet_local_id.to_numpy(np.int64),f)
    i = pairs(sg,f,30,30)
    origin = pd.DataFrame(dict(frame=f[i],x=p[i,0],y=p[i,1],
                              speed=np.linalg.norm(p[i+30]-p[i],axis=1)/(30*DT)))
    for frame,d in origin.groupby('frame'):
        q = clouds['land'][int(frame)//30]
        if len(q)<2: continue
        nearest = cKDTree(q).query(d[['x','y']].to_numpy(),k=2)[0][:,1]
        for distance,speed in zip(nearest,d.speed):
            density_speed.append((int(frame),distance,speed))
    ds = pd.DataFrame(density_speed,columns=['frame','nearest_px','speed_px_s'])
    ds.to_csv(out/'tables'/f'{video}_density_speed_samples.csv',index=False)
    bins = [0,50,100,200,400,800,np.inf]
    br = []
    for a,b in zip(bins[:-1],bins[1:]):
        v = ds.loc[(ds.nearest_px>=a)&(ds.nearest_px<b),'speed_px_s'].to_numpy()
        if len(v): br.append(dict(r_lo=a,r_hi=b,n=len(v),median=np.median(v),
                                 q25=np.quantile(v,.25),q75=np.quantile(v,.75),mean=v.mean()))
    pd.DataFrame(br).to_csv(out/'tables'/f'{video}_density_speed.csv',index=False)
    # Maps, including static water polygon for interpretation.
    polygon = np.array(json.loads((DATA/'config/water_regions.json').read_text())['videos'][video]['polygon_xy'])
    fig,axs = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,pop in zip(axs,['all','land']):
        mean = grids[pop].mean(axis=0)
        im = ax.imshow(mean,extent=(0,W,H,0),norm=LogNorm(vmin=.001,vmax=max(1,mean.max())),cmap='magma')
        ax.plot(*np.vstack([polygon,polygon[0]]).T,color='cyan',lw=.8)
        ax.set(title=f'{pop}: mean detections per 320 x {H/14:.1f} px cell',xlabel='x (px)',ylabel='y (px)')
        fig.colorbar(im,ax=ax,shrink=.7,label='detections / sampled frame / cell')
    fig.suptitle(video+' | spatial occupancy (not burrow positions)')
    savefig(out,f'{video}_occupancy',fig)
    np.savez_compressed(out/'tables'/f'{video}_density_grids.npz',frames=frames,**grids)
    return summaries


def analyze_video(video,out,rng):
    print('Loading',video,flush=True)
    cols = ['frame','tracklet_local_id','accepted_for_tracking','body_x','body_y','body_score','water']
    df = pd.read_parquet(DATA/'tables/observations'/video,columns=cols)
    assert df.frame.between(0,NF-1).all()
    df['accepted_for_tracking'] = df.accepted_for_tracking.fillna(False).astype(bool)
    ok = df.accepted_for_tracking
    assert df.loc[ok,'tracklet_local_id'].notna().all()
    xy = df[['body_x','body_y']].to_numpy(float)
    valid = np.isfinite(xy).all(axis=1)&(xy[:,0]>=0)&(xy[:,0]<W)&(xy[:,1]>=0)&(xy[:,1]<H)
    assert valid[ok].all()
    counts = pd.DataFrame({'frame':np.arange(NF),'time_s':np.arange(NF)*DT})
    masks = {'raw':np.ones(len(df),bool),'finite_in_image':valid,'accepted':ok.to_numpy(),
             'land':(ok & df.water.eq(False)).to_numpy(bool),
             'water':(ok & df.water.eq(True)).to_numpy(bool)}
    for name,mask in masks.items(): counts[name] = np.bincount(df.loc[mask,'frame'],minlength=NF)
    reference = pd.read_parquet(DATA/'tables/frames.parquet')
    reference = reference.loc[reference.video_id.eq(video)].sort_values('frame')
    assert np.array_equal(reference.frame, counts.frame)
    assert np.array_equal(reference.source_count, counts.raw)
    assert np.array_equal(reference.assigned_count, counts.accepted)
    assert np.array_equal(counts.land + counts.water, counts.accepted)
    counts.to_csv(out/'tables'/f'{video}_counts.csv',index=False)
    summary = {'raw_detections':len(df),'accepted_detections':int(ok.sum()),
               'mean_counts':{k:float(counts[k].mean()) for k in masks},
               'land_first_minute':float(counts.land.iloc[:1800].mean()),
               'land_last_minute':float(counts.land.iloc[-1800:].mean())}
    summary['spatial'] = spatial(df,video,out)
    summary['motion'] = {}
    for pop in ['all','land']:
        print('Motion',video,pop,flush=True)
        _,_,_,ss = motion(df,video,pop,out,rng)
        summary['motion'][pop] = ss
    # Track length distribution is identity QC, not biological dwell time.
    lengths = df.loc[ok].groupby('tracklet_local_id').size().to_numpy()
    lens,n = np.unique(lengths,return_counts=True)
    pd.DataFrame(dict(frames=lens,tracklets=n)).to_csv(out/'tables'/f'{video}_track_lengths.csv',index=False)
    return summary


def combined_figures(out):
    colors = ['#1675ac','#c85322']
    fig,axs = plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for ax,v in zip(axs,VIDEOS):
        d = pd.read_csv(out/'tables'/f'{v}_counts.csv')
        z = d.groupby(np.arange(len(d))//300).mean(numeric_only=True)
        for col in ['raw','accepted','land','water']:
            ax.plot(z.time_s/60,z[col],label=col)
        ax.set(ylabel='mean detections / frame',title=v); ax.legend(ncol=4,fontsize=8)
    axs[-1].set_xlabel('nominal video time (min)'); savefig(out,'counts',fig)
    fig,axs = plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    for v,c in zip(VIDEOS,colors):
        d = pd.read_csv(out/'tables'/f'{v}_track_lengths.csv')
        axs[0].step(d.frames*DT,np.cumsum(d.tracklets[::-1])[::-1]/d.tracklets.sum(),where='post',label=v,color=c)
        m = pd.read_csv(out/'tables'/f'{v}_land_motion.csv')
        axs[1].semilogx(m.lag_s,m.origin_survival,'o-',label=v,color=c)
    axs[0].set(xscale='log',yscale='log',xlabel='tracklet length x frame interval (s)',ylabel='fraction of tracklets at least this long')
    axs[1].set(xlabel='displacement lag (s)',ylabel='fraction of land origins still eligible',ylim=(0,1))
    for ax in axs: ax.legend(fontsize=8); ax.grid(alpha=.2)
    savefig(out,'tracking_support',fig)
    fig,axs = plt.subplots(2,2,figsize=(10,7.2),layout='constrained')
    for v,c in zip(VIDEOS,colors):
        for pop,ls in [('land','-'),('all','--')]:
            d = pd.read_csv(out/'tables'/f'{v}_{pop}_motion.csv')
            label=v.replace('Crab_','')+' '+pop
            axs[0,0].loglog(d.lag_s,d.msd_px2,ls,color=c,label=label)
            if pop=='land':
                axs[0,0].fill_between(d.lag_s,d.msd_block_low,d.msd_block_high,color=c,alpha=.15)
                axs[0,1].semilogx(d.lag_s,d.alpha2,'o-',color=c,label=label)
                axs[1,0].semilogx(d.lag_s,d.overlap_10px,'o-',color=c,label=label)
                axs[1,1].loglog(d.lag_s,d.msd_px2,'o-',color=c,label=label+' lag-specific')
                axs[1,1].loglog(d.lag_s,d.fixed_origin_msd_px2,':',color=c,label=label+' fixed 10 s support')
                axs[1,1].loglog(d.lag_s,d.high_score_msd_px2,'--',color=c,label=label+' endpoint score >=0.8')
    axs[0,0].set(xlabel='lag (s)',ylabel='MSD (px squared)',title='Conditional motion; shaded block resampling range')
    axs[0,1].set(xlabel='lag (s)',ylabel='2D non-Gaussian parameter',title='alpha2 = <r^4> / (2 <r^2>^2) - 1')
    axs[0,1].axhline(0,color='gray',lw=.8)
    axs[1,0].set(xlabel='lag (s)',ylabel='P(displacement < 10 px)',ylim=(0,1),title='Overlap / localization diagnostic')
    axs[1,1].set(xlabel='lag (s)',ylabel='MSD (px squared)',title='Selection sensitivity (land only)')
    for ax in axs.flat: ax.legend(fontsize=6); ax.grid(alpha=.2)
    savefig(out,'motion',fig)
    fig,axs = plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,v in zip(axs,VIDEOS):
        h = pd.read_csv(out/'tables'/f'{v}_land_vanhove.csv')
        m = pd.read_csv(out/'tables'/f'{v}_land_motion.csv').set_index('lag_frames')
        for lag,c in zip([3,30,300],['#1675ac','#c85322','#39954c']):
            z=h.loc[h.lag_frames==lag]; r=(z.r_lo+z.r_hi)/2; pdf=z.radial_pdf
            ax.loglog(r,pdf,color=c,label=f'{lag*DT:.2f} s')
            rr=np.geomspace(.05,2000,300); msd=m.loc[lag,'msd_px2']
            ax.loglog(rr,2*rr/msd*np.exp(-rr*rr/msd),'--',color=c,alpha=.6)
        ax.set(xlim=(.1,2000),ylim=(1e-7,10),xlabel='radial displacement r (px)',ylabel='radial PDF p(r; lag) (1/px)',title=v+' | land')
        ax.legend(); ax.grid(alpha=.2)
    savefig(out,'vanhove',fig)
    fig,axs = plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for v,c in zip(VIDEOS,colors):
        d=pd.read_csv(out/'tables'/f'{v}_all_pairs.csv'); r=(d.r_lo+d.r_hi)/2
        axs[0].plot(r,d.g_rectangle,color=c,label=v)
        d=pd.read_csv(out/'tables'/f'{v}_land_pairs.csv')
        axs[1].plot(r,d.same_to_time_mixed,color=c,label=v)
        d=pd.read_csv(out/'tables'/f'{v}_land_nn.csv')
        axs[2].plot(d.r_hi,d.cdf,color=c,label=v)
    axs[0].set(xlabel='pair separation (px)',ylabel='g(r)',title='All: rectangle translation correction')
    axs[1].set(xlabel='pair separation (px)',ylabel='same-time / time-mixed pair density',title='Land: empirical spatial baseline')
    axs[2].set(xlabel='nearest-neighbor distance (px)',ylabel='cumulative fraction',xlim=(0,1000),title='Land: nearest detected neighbor')
    for ax in axs[:2]: ax.axhline(1,color='gray',lw=.8); ax.set_xlim(0,2000)
    for ax in axs: ax.legend(fontsize=7); ax.grid(alpha=.2)
    savefig(out,'spatial_pairs',fig)
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for v,c in zip(VIDEOS,colors):
        a=pd.read_csv(out/'tables'/f'{v}_density_acf.csv'); a=a[a.population=='land']
        axs[0].plot(a.lag_s,a.count_acf,color=c,label=v+' raw')
        axs[0].plot(a.lag_s,a.detrended_count_acf,'--',color=c,label=v+' detrended')
        axs[1].plot(a.lag_s,a.cell_acf,color=c,label=v+' connected')
        axs[1].plot(a.lag_s,a.shape_acf,'--',color=c,label=v+' count-adjusted')
        a=pd.read_csv(out/'tables'/f'{v}_land_vacf.csv')
        axs[2].plot(a.lag_s,a.connected_vacf,'o-',color=c,label=v)
    axs[0].set(xlabel='lag (s)',ylabel='count autocorrelation',title='Land count fluctuations')
    axs[1].set(xlabel='lag (s)',ylabel='cell-density autocorrelation',title='Land density memory (24 x 14 cells)')
    axs[2].set(xlabel='lag (s)',ylabel='connected velocity correlation',title='Velocity averaged over 0.334 s')
    for ax in axs: ax.axhline(0,color='gray',lw=.8); ax.legend(fontsize=6); ax.grid(alpha=.2)
    savefig(out,'correlations',fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for ax,v in zip(axs,VIDEOS):
        d=pd.read_csv(out/'tables'/f'{v}_density_speed.csv')
        xx=np.arange(len(d)); labels=[f'{a:g}-{b:g}' for a,b in zip(d.r_lo,d.r_hi)]
        ax.errorbar(xx,d['median'],yerr=[d['median']-d.q25,d.q75-d['median']],fmt='o',capsize=4,label='median and IQR (not uncertainty)')
        for x,y,n in zip(xx,d.q75,d.n): ax.annotate(f'n={n:,}',(x,y),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7)
        ax.set_xticks(xx,labels,rotation=25); ax.set(xlabel='origin nearest land detection (px)',ylabel='1.001-s displacement / time (px/s)',title=v)
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    savefig(out,'density_speed',fig)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Computations must run under Slurm'
    start=time.time(); out=BASE/f"run_{os.environ['SLURM_JOB_ID']}"
    out.mkdir(exist_ok=False); (out/'figures').mkdir(); (out/'tables').mkdir()
    unit_receipt=json.loads((BASE/f"unit_tests_{os.environ['SLURM_JOB_ID']}.json").read_text())
    assert unit_receipt['passed'] and unit_receipt['tests_run'] >= 16
    test_result=tests(); test_result['unit_tests']=unit_receipt
    dump(out/'tests.json',test_result)
    complete=json.loads((DATA/'DATASET_COMPLETE.json').read_text())
    assert complete['status']=='validated_complete'
    manifest_hash=sha(DATA/'manifest.json'); assert manifest_hash==complete['manifest_sha256']
    manifest=json.loads((DATA/'manifest.json').read_text())
    pins=[]
    for artifact in manifest['artifacts']:
        if artifact['path'].startswith('tables/observations/') or artifact['path'] in ['tables/frames.parquet','tables/tracklets.parquet','config/water_regions.json']:
            path=DATA/artifact['path']; digest=sha(path)
            assert digest==artifact['sha256'],path
            pins.append({'path':artifact['path'],'sha256':digest})
    print('Verified',len(pins),'input artifacts',flush=True)
    rng=np.random.default_rng(SEED)
    summaries={v:analyze_video(v,out,rng) for v in VIDEOS}
    assert sum(s['raw_detections'] for s in summaries.values())==complete['counts']['observations']
    assert sum(s['accepted_detections'] for s in summaries.values())==complete['counts']['assigned_observations']
    dump(out/'summary.json',summaries)
    combined_figures(out)
    provenance={'job_id':os.environ['SLURM_JOB_ID'],'dataset_manifest_sha256':manifest_hash,
                'inputs':pins,'analysis_sha256':sha(Path(__file__)),
                'report_code_sha256':sha(BASE/'report.py'),'job_script_sha256':sha(BASE/'run.sh'),
                'unit_test_code_sha256':sha(BASE/'test_analysis.py'),
                'python':sys.version,'platform':platform.platform(),
                'versions':{m.__name__:m.__version__ for m in [np,pd,scipy,matplotlib]},
                'seed':SEED,'frame_interval_s':DT,'spatial_stride_frames':30,
                'motion_origin_stride_frames':10,'lags_frames':LAGS,
                'elapsed_before_report_s':time.time()-start}
    dump(out/'provenance.json',provenance)
    from report import make_report
    make_report(out)
    artifacts=[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)}
               for p in sorted(out.rglob('*')) if p.is_file()]
    dump(out/'COMPLETE.json',{'status':'complete','job_id':os.environ['SLURM_JOB_ID'],
                             'elapsed_s':time.time()-start,'artifacts':artifacts})
    print('COMPLETE',out,flush=True)


if __name__=='__main__': main()
