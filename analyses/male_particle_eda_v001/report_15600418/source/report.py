"""Technical PDF and matching Markdown narrative, generated from saved results."""
from pathlib import Path
import json, textwrap, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

VIDEOS=['Crab_Cam_2_2','Crab_Cam_2_3']
DT=1001/30000


def make_report(out, destination=None):
    out=Path(out)
    destination=Path(destination) if destination is not None else out
    destination.mkdir(exist_ok=True)
    summary=json.loads((out/'summary.json').read_text())
    prov=json.loads((out/'provenance.json').read_text()); tests=json.loads((out/'tests.json').read_text())
    motions={v:pd.read_csv(out/'tables'/f'{v}_land_motion.csv').set_index('lag_frames') for v in VIDEOS}
    pages=destination/'report_pages'; pages.mkdir(exist_ok=True)
    narrative=[]; page_no=0
    pdfpath=destination/'male_crab_preliminary_report.pdf'
    with PdfPages(pdfpath,metadata={'Title':'Male fiddler crab particle-system EDA',
                                   'Author':'Preliminary computational analysis',
                                   'Subject':'Detection structure and experimental tracklet dynamics'}) as pdf:
        def finish(fig):
            nonlocal page_no
            page_no+=1
            fig.text(.94,.025,str(page_no),ha='right',fontsize=8,color='gray')
            pdf.savefig(fig)
            fig.savefig(pages/f'page_{page_no:02}.png',dpi=100)
            plt.close(fig)

        def textpage(title, paragraphs):
            narrative.append('# '+title+'\n\n'+'\n\n'.join(paragraphs))
            fig=plt.figure(figsize=(8.27,11.69)); fig.patch.set_facecolor('white')
            fig.text(.085,.94,title,fontsize=17,weight='bold',va='top')
            y=.885
            for paragraph in paragraphs:
                lines=[]
                for line in paragraph.split('\n'):
                    lines.extend(textwrap.wrap(line,width=91,break_long_words=False,break_on_hyphens=False) or [''])
                for line in lines:
                    fig.text(.085,y,line,fontsize=10,va='top',fontfamily='DejaVu Sans')
                    y-=.019
                y-=.017
            assert y>.045, (title,y)
            finish(fig)

        def figurepage(title, name, caption):
            image_relative=os.path.relpath(out/'figures'/f'{name}.png',destination)
            narrative.append('# '+title+'\n\n'+caption+f'\n\n![{title}]({image_relative})')
            fig=plt.figure(figsize=(11.69,8.27)); fig.patch.set_facecolor('white')
            fig.text(.055,.95,title,fontsize=17,weight='bold',va='top')
            ax=fig.add_axes([.035,.22,.93,.64]); ax.imshow(plt.imread(out/'figures'/f'{name}.png')); ax.axis('off')
            lines=textwrap.wrap(caption,width=139,break_long_words=False)
            assert len(lines)<=8,(title,len(lines))
            fig.text(.06,.185,'\n'.join(lines),fontsize=9.5,va='top',linespacing=1.45)
            finish(fig)

        facts=[]
        for v in VIDEOS:
            s=summary[v]; m=motions[v]
            facts.append(f"{v}: {s['mean_counts']['land']:.1f} accepted land detections/frame on average; "
                         f"nearest-neighbor median {s['spatial']['land']['nn_quantiles_px'][1]:.1f} px. "
                         f"At 1.001 s, displacement median {m.loc[30,'median_r_px']:.2f} px, "
                         f"RMS {m.loc[30,'rms_px']:.2f} px, alpha2 {m.loc[30,'alpha2']:.2f}. "
                         f"At 10.01 s, only {100*m.loc[300,'origin_survival']:.1f}% of sampled land origins "
                         f"retain a usable uninterrupted path.")
        textpage('Male fiddler crabs: preliminary particle EDA',[
            'Scope: two videos, 28,800 frames each, 7,985,830 source detections; nominal sampling 29.970 Hz. '
            'Each movie spans 960.96 s including the final frame interval. No females, burrows, physical '
            'distance calibration, or validated long-term animal identities are available.',
            'This report separates spatial statistics of accepted male-targeted detections from motion '
            'statistics conditional on experimental tracklet continuity. Coordinates are Shell landmarks '
            '(explicit centroid proxies) in 7680 x 4320 original-image pixels, not physical body centers. '
            'Results describe this model output, not a census of uniquely identified males.',
            *facts,
            'Main interpretation: spatial occupancy and short-time displacement distributions are accessible '
            'with male data alone. Non-Gaussian displacement statistics, count/density memory and spatial '
            'pair structure are useful diagnostics, but neither social forces, territorial organization, '
            'nor biological diffusion can be inferred from these summaries alone. Localization jitter, '
            'persistent false detections, common image motion and survivor selection remain alternatives.',
            f"Reproducibility: CPU Slurm job {prov['job_id']}; fixed seed {prov['seed']}; "
            f"{tests['unit_tests']['tests_run']} known-answer unit tests passed before the real-data analysis, "
            'plus input and output validation. Source tables, predictions, trackers and models were not modified.'
        ])
        selection_facts=[]
        for v in VIDEOS:
            m=motions[v]
            selection_facts.append(f"{v}: restricting to the same origins surviving 10.01 s lowers "
                                   f"the 1.001-s MSD by {100*(1-m.loc[30,'fixed_origin_msd_px2']/m.loc[30,'msd_px2']):.1f}%; "
                                   f"the >=0.8 endpoint-score check changes it by "
                                   f"{100*(m.loc[30,'high_score_msd_px2']/m.loc[30,'msd_px2']-1):+.1f}%.")
        textpage('Findings from the reviewed plots',[
            '1. Heterogeneous apparent mobility. The measured radial van Hove densities contain a '
            'large near-zero component and a much broader tail than the matched-MSD Gaussian '
            'benchmark. At 1.001 s, 90.7% / 82.7% of land pairs move <10 px in Cam 2_2 / Cam 2_3. '
            'The strong positive alpha2 values (9.80 / 5.13) are inconsistent with that simple '
            'Gaussian description of the observed distributions; they identify no specific '
            'biological mechanism.',
            f"2. Larger displacements in Cam 2_3. Its land MSD is "
            f"{motions[VIDEOS[1]].loc[30,'msd_px2']/motions[VIDEOS[0]].loc[30,'msd_px2']:.2f} times "
            'Cam 2_2 at 1.001 s. This difference persists under the fixed-origin and endpoint-score '
            'checks. It is a difference between these recorded/model-derived samples, not a '
            'population-level comparison. Survival selection is more consequential than the '
            'endpoint-score threshold here. '+ ' '.join(selection_facts),
            '3. Spatial concentration largely follows the occupancy footprint. Rectangle-corrected '
            'g(r) is elevated across much of 0-2000 px, whereas the land same-time/time-mixed '
            'ratio is visually close to one beyond roughly 200 px. At 50-100 px the ratios are '
            '0.61 / 0.70; at 100-200 px they are 0.95 / 0.94. This contrast warns against '
            'calling the uniform-null clustering social attraction. The short-distance deficit '
            'also cannot establish avoidance because time-mixed self-persistence is unresolved.',
            '4. Persistent dynamics, but nonstationary counts. Connected velocity correlations '
            'remain positive: 0.43 / 0.51 at 1.001 s and 0.13 / 0.21 at 10.01 s. Connected '
            'coarse-cell density correlations decay over tens of seconds; removing total-count '
            'variation changes them little. The large long-lag count correlation in Cam 2_3 '
            'is strongly reduced by linear detrending. Count memory is therefore not a clean '
            'collective relaxation time.',
            '5. Spacing-speed and resolution warnings. More isolated detections have lower median '
            '1-s apparent speed in the plotted bins, but overlapping IQRs, spatial confounding '
            'and encounter-related track loss preclude a causal conclusion. Both van Hove plots '
            'show a shared feature around the one-pixel scale; this deserves a localization / '
            'prediction-discretization check, not interpretation as a preferred biological step.'
        ])
        textpage('Methods and estimands',[
            'Selections. Raw counts retain every detection. Spatial and motion primary selections use '
            'accepted_for_tracking=True (finite, in-image Shell; body score >=0.5; duplicate-candidate rejection). '
            'Land means the accepted fixed-polygon water label is explicitly false, not missing. All accepted '
            'detections are retained as a sensitivity comparison. Land removal splits continuous segments; '
            'no displacement crosses a gap, water interval, video boundary or tracklet ID change.',
            'Sampling. Counts use every frame. Spatial statistics use frames 0,30,...,28770 (960 frames/movie; '
            '1.001-s spacing). Motion uses time origins at original frame multiples of 10, with lags '
            '1,3,10,30,60,150,300,600,1500 frames (0.0334 to 50.05 s). Every retained origin-endpoint path '
            'must remain continuously observed. All measurements are unsmoothed. A fixed-origin comparison '
            'holds the eligible origins at the 300-frame lag constant for all shorter lags.',
            'Motion definitions. MSD(tau) = mean |x(t+tau)-x(t)|^2. The self van Hove function is shown as '
            'the radial probability density p(r;tau) = 2*pi*r*G_s(r;tau) under angular averaging, not G_s '
            'per unit area. Histograms include zero displacement and integrate to one. The isotropic '
            'zero-drift Gaussian reference is p(r) = 2r/MSD * exp(-r^2/MSD). '
            'alpha2 = mean(r^4)/(2*mean(r^2)^2)-1; alpha2=0 for that Gaussian. '
            'Q(10 px,tau)=P(r<10 px). Drift is not subtracted from displacement moments.',
            'Uncertainty. MSD bands are the 2.5th/97.5th percentiles of 300 pooled-ratio resamples of '
            '60-s origin-time blocks (16 full blocks and a short final block). These are within-video '
            'stability diagnostics, not independent-animal confidence intervals. Long-range dependence, '
            'track selection, systematic model error and the short final block limit interpretation. '
            'Other plots are descriptive; no p-values, multiple-testing claims, diffusion fits or '
            'population-level inference are made.',
            'Velocity. Coarse velocity is [x(t+10)-x(t)]/(10*dt). Reported correlations use centered '
            'velocity vectors and an RMS-product denominator on the SAME surviving origin pairs for '
            'each lag. Intervals do not overlap at positive tested lags (>=10 frames). This reduces, '
            'but does not eliminate, localization-noise sensitivity. Raw correlations are also saved.'
        ])
        figurepage('1. Identity support is the first limitation','tracking_support',
            'Tracklet survival is not animal lifespan. Most tracklets are short even though long tracks '
            'contain many observations. Right: eligible land origin-endpoint pairs divided by all sampled '
            'land origins, including origins too close to the end of a video; thus both fragmentation '
            'and recording-end censoring reduce support. Long-lag motion is strongly conditional on '
            'remaining detected and associated. The tracker forbids single-frame links >12 px, uses '
            'velocity prediction, and rejects ambiguous neighbors, so motion and close encounters are selected.')
        figurepage('2. Detection counts and nonstationarity','counts',
            'Ten-second display averages; raw calculations use every original frame, including zeros. '
            'Raw, accepted and land curves may overlap because relatively few rows are unassigned or '
            'water-labeled. Counts are detections, not unique males or population estimates. '
            + ' '.join(f"{v}: land first/last nominal minute {summary[v]['land_first_minute']:.1f}/"
                       f"{summary[v]['land_last_minute']:.1f} detections per frame. " for v in VIDEOS)
            + 'Changing detection conditions or activity can alter these series without abundance changes.')
        for v in VIDEOS:
            figurepage('3. Spatial occupancy: '+v,v+'_occupancy',
                'Time-averaged counts on a fixed 24 x 14 grid; each cell is 320 x 308.6 image pixels. '
                'Cyan outlines the accepted static water polygon. Color is log-scaled detections per '
                'sampled frame per cell, NOT density per square meter or per unit usable habitat. '
                'Zero cells are masked. Concentrations can reflect habitat, detection performance or '
                'persistent artifacts. Similar spatial maps in the videos do not establish different '
                'animals or independent biological replicates. No hotspot is identified as a burrow.')
        figurepage('4. Displacement scaling and selection sensitivity','motion',
            'MSD is conditional on uninterrupted experimental identity and is not fitted to a diffusion '
            'law. Solid/dashed comparisons separate land and all accepted detections. Alpha2 and the '
            '10-pixel overlap quantify broad displacement distributions and localization. Bottom-right '
            'holds origins fixed to paths surviving 10.01 s (dotted) and separately checks >=0.8 endpoint '
            'scores (dashed; not a full-path confidence filter). Agreement would not establish identity '
            'accuracy; disagreement diagnoses selection. Shading is a temporal-block resampling range.')
        figurepage('5. Self van Hove radial displacement distributions','vanhove',
            'Solid curves are radial displacement probability densities for land segments; dashed curves '
            'are isotropic zero-drift Gaussian/Rayleigh references with the same measured MSD. '
            'Normalization includes the zero-to-0.1-pixel bin even though the displayed log axis starts '
            'at 0.1 px. The plot is p(r), not the area-normalized G_s. Differences can arise from mixed '
            'mobilities, pausing, drift, localization error and tracking selection. Heavy tails do not '
            'by themselves demonstrate anomalous diffusion or behavioral switching.')
        spatialfacts=' '.join(f"{v}: median nearest land detection "
                              f"{summary[v]['spatial']['land']['nn_quantiles_px'][1]:.1f} px; "
                              f"{100*summary[v]['spatial']['land']['fraction_nn_under_100']:.1f}% "
                              'have a neighbor <100 px.' for v in VIDEOS)
        figurepage('6. Pair-distance structure and nearest neighbors','spatial_pairs',
            'Left: full-rectangle translation-corrected g(r), whose uniform independent null is one; '
            'the actual habitat is not uniform. Middle: same-time land pair-distance probability '
            'divided by a time-mixed baseline from offsets of 60,120,240 sampled frames (circular), '
            'each normalized by its TOTAL possible pairs, not only plotted radii. Right: nearest-neighbor '
            'CDF without boundary correction (descriptive only). '+spatialfacts)
        textpage('Interpreting spatial correlation without burrows',[
            'The full-image pair estimate uses unordered distinct pairs and weights each separation '
            '(dx,dy) by W*H/[(W-|dx|)*(H-|dy|)]. Weighted annular counts are divided by the sum of '
            'N(N-1)/2 over sampled frames, times annular area/(W*H). This corrects rectangular '
            'observation-window truncation, not inaccessible habitat, perspective or spatially '
            'varying detectability. A uniform-rectangle control verifies the normalization.',
            'The time-mixed land baseline compares positions in different sampled frames, using cross '
            'pairs normalized by sum N(t)*N(t+offset). It preserves the empirical footprint rather '
            'than assuming that the entire land area is accessible. Offsets are approximately '
            '60.06, 120.12 and 240.24 s with circular wrapping. Normalization is pair-weighted across '
            'frames. It is an exploratory diagnostic, not an inhomogeneous-point-process estimate '
            'or a hypothesis-test null.',
            'Crucially, time-mixed pairs can contain the SAME stationary male, repeated detections '
            'at a burrow, or a persistent artifact at both times. Those cannot be reliably removed '
            'using fragmented IDs. A short-distance deficit against this baseline could therefore '
            'be self-persistence rather than interpersonal avoidance. Nonstationarity also changes '
            'the baseline. The rectangle and time-mixed plots should be read together.',
            'At very short distances, the tracking acceptance rule rejects bodies within 5 pixels '
            'as duplicate candidates. Missed close individuals, unresolved body overlap and tracker '
            'ambiguity add selection. Nearest-neighbor distances near image borders may be '
            'overestimated because neighbors outside the image are unseen. Neither a preferred '
            'spacing nor a repulsion scale can yet be assigned a biological cause.',
            'The most useful next spatial additions are a usable-habitat mask, physical/perspective '
            'calibration, explicit burrow coordinates and representative detection audits. Female '
            'positions would enable male-female cross-correlations; they are not necessary for '
            'these descriptive male-only statistics, but are necessary for courtship-specific claims.'
        ])
        figurepage('7. Temporal count, density and velocity memory','correlations',
            'Count ACF subtracts the movie mean; dashed also removes a linear time trend. Cell ACF '
            'subtracts each cell mean and sums covariance over 24 x 14 cells. The count-adjusted '
            'version subtracts N(t) times the mean occupancy profile before centering, separating '
            'spatial redistribution from total-count changes. ACF uses available pairs at each lag '
            'and lag-zero variance. Velocity correlations use continuously tracked land segments. '
            'Positive memory need not be social coordination; persistent occupancy and detection '
            'conditions also generate it. Adjacent velocity increments share endpoint localization noise.')
        figurepage('8. Local spacing versus apparent speed','density_speed',
            'For accepted land detections at sampled frames, the closest OTHER accepted land '
            'detection is found at the origin; speed is the 30-frame displacement / 1.001 s on '
            'a surviving land segment. Points show medians, bars the interquartile range, not '
            'confidence intervals; n counts correlated origin pairs, not animals. Bins are fixed '
            'in image pixels and the final bin is unbounded. Any trend is descriptive and may '
            'reflect habitat, perspective, encounter-related fragmentation or misdetections, '
            'not a causal effect of density on behavior.')
        for v in VIDEOS:
            figurepage('9. Trajectory examples: '+v,v+'_trajectories',
                'Twenty longest continuous land segments, deliberately selected for visibility; '
                'points mark their starts. Displayed vertices are sampled every 10 frames, without '
                'smoothing or gap bridging in the analysis. Axes are absolute image coordinates '
                'with y increasing downward. These examples are not representative random animals '
                'and have not been validated against video ground truth. Strong localization may '
                'reflect site fidelity, immobility, limited duration or a persistent false detection.')
        quant=[]
        for v in VIDEOS:
            m=motions[v]; s=summary[v]
            quant.append(f"{v}: at 0.1001 / 1.001 / 10.01 s, land MSD = "
                         + ' / '.join(f"{m.loc[k,'msd_px2']:.2f}" for k in [3,30,300])
                         + ' px^2; alpha2 = '+ ' / '.join(f"{m.loc[k,'alpha2']:.2f}" for k in [3,30,300])+'. '
                         +f"At 1.001 s, {100*m.loc[30,'overlap_10px']:.1f}% of retained displacement pairs "
                         +f"are <10 px ({int(m.loc[30,'n_pairs']):,} origin pairs; "
                         +f"{int(m.loc[30,'n_segments']):,} contributing segments). "
                         +f"The all-accepted one-frame step maximum is {s['motion']['all']['one_frame_step_quantiles'][-1]:.4f} px.")
        textpage('Quantitative synopsis and limits',[
            *quant,
            'The Gaussian reference is a benchmark, not a fitted mechanistic model. Nonzero drift '
            'also changes alpha2 relative to a centered isotropic Gaussian. Pixel localization '
            'noise contributes an MSD plateau: independent per-coordinate error variance sigma^2 '
            'produces MSD=4*sigma^2 even for a stationary body. Motion near this scale cannot '
            'be interpreted as locomotion without a localization-error calibration.',
            'Longer paths select easier-to-track, often more stationary detections. The tracker '
            'also imposes a 12-pixel single-frame step ceiling (~359.6 px/s), a predicted-position '
            'gate, matching costs and ambiguity checks. Thus the observed high-speed tail and '
            'close-encounter dynamics are not unbiased. Endpoint score sensitivity is not '
            'calibration: model scores are not probabilities of a correct detection.',
            'No female interaction, burrow attachment, mating behavior, true animal abundance, '
            'diffusion coefficient, collective phase or social attraction/repulsion is claimed. '
            'The two files are analyzed separately and are not assumed to be independent fields '
            'or independent male populations. No cross-video identity linkage is attempted.'
        ])
        textpage('Tests, validation and reproducibility',[
            f"All {tests['unit_tests']['tests_run']} unit tests passed before analysis. Known-answer "
            'controls cover stationary and constant-velocity trajectories; Gaussian diffusion '
            '(MSD=8*lag for component step variance 4); localization-noise plateau; gap and '
            'identity isolation; water-filter splitting; fixed-origin support; a PDF atom at zero '
            'and Rayleigh CDF; exact three-point pair/nearest-neighbor geometry; uniform-rectangle '
            'g(r) and independent time-mixed ratio; white-noise and AR(1) ACFs; cell mean '
            'invariance; constant-value block resampling; rotational invariance; invalid data.',
            'Real-data checks: completion marker and manifest checksum; full SHA-256 comparison '
            'of every consumed observation part, frame and tracklet table, and water geometry '
            'against the published manifest; exact frame-level raw/assigned count reconciliation; '
            'assigned coordinate/ID validity; consecutive original frames within IDs; land/water '
            'count accounting; global row accounting; no gap pairs; all one-frame steps <=12 px; '
            'all displacement histogram counts retained. Checks fail the job rather than '
            'silently filling gaps, clipping positions, changing tracking or dropping bad outputs.',
            'Software checks establish estimator correctness and internal consistency, NOT '
            'detection accuracy or true identity. Controls are synthetic and do not capture '
            'structured field-video errors. No random identity review or calibration set was '
            'available for this analysis.',
            f"Run directory: analyses/male_particle_eda_v001/run_{prov['job_id']}/. "
            'summary.json gives headline quantities; tables/ contains numerical curves and '
            'density grids; figures/ contains PNG/PDF plots; report_pages/ contains page previews; '
            'tests.json records the validation receipt; provenance.json pins inputs, code and '
            'package versions. COMPLETE.json lists output checksums after successful report creation.',
            'Dataset: datasets/v007_best_body_first_dataframe_v001/. Canonical schema/method '
            'references: docs/datasets/v007_best_body_first_dataframe_v001/README.md and '
            'docs/BODY_FIRST_TRACKLETS.md. Analysis source and finite Slurm entry point are '
            'analysis.py, report.py, test_analysis.py and run.sh in the analysis directory. '
            'This is a preliminary, reproducible diagnostic report rather than a validated '
            'biological inference.'
        ])
        expected_pages=pdf.get_pagecount()
    (destination/'report.md').write_text('\n\n---\n\n'.join(narrative)+'\n')
    # Read-back of image artifacts and basic PDF framing; no claim of full PDF parsing.
    assert expected_pages==page_no and page_no>=12
    for path in (out/'figures').glob('*.png'):
        with Image.open(path) as im: im.verify()
    content=pdfpath.read_bytes()
    assert content.startswith(b'%PDF-') and b'%%EOF' in content[-1024:]
    (destination/'report_validation.json').write_text(json.dumps({
        'pages':page_no,'pdf_bytes':len(content),'pdf_header_and_eof_valid':True,
        'figure_png_readback_valid':True,'narrative_layout_bounds_passed':True},indent=2)+'\n')


if __name__=='__main__':
    import os,sys
    assert os.environ.get('SLURM_JOB_ID'), 'Report generation must run in Slurm'
    make_report(Path(sys.argv[1]))
