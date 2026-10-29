"""Review-stage output readback and report only; never reruns the real-data estimators."""
from pathlib import Path
import os, json, shutil, subprocess
import numpy as np
import pandas as pd
from analysis import sha, dump, VIDEOS
from report import make_report


def main():
    job=os.environ['SLURM_JOB_ID']
    base=Path(__file__).resolve().parent
    source=base/'run_15600144'
    completion=json.loads((source/'COMPLETE.json').read_text())
    assert completion['status']=='complete'
    for artifact in completion['artifacts']:
        p=source/artifact['path']
        assert p.stat().st_size==artifact['bytes'] and sha(p)==artifact['sha256'],p
    checks=[]
    for video in VIDEOS:
        for pop in ['all','land']:
            m=pd.read_csv(source/'tables'/f'{video}_{pop}_motion.csv')
            assert np.isfinite(m.msd_px2).all() and (m.msd_px2>=0).all()
            assert m.origin_survival.between(0,1).all()
            assert m.overlap_10px.between(0,1).all()
            assert np.all(np.diff(m.n_pairs)<=0)
            assert (m.high_score_pairs<=m.n_pairs).all()
            fixed=m.loc[m.lag_frames<=300]
            assert fixed.fixed_origin_pairs.nunique()==1
            last=fixed.iloc[-1]
            assert last.msd_px2==last.fixed_origin_msd_px2
            h=pd.read_csv(source/'tables'/f'{video}_{pop}_vanhove.csv')
            for lag,z in h.groupby('lag_frames'):
                assert z['count'].sum()==z.n_pairs.iloc[0]
                assert np.isclose(np.sum(z.radial_pdf*(z.r_hi-z.r_lo)),1)
                assert z.n_pairs.iloc[0]==m.set_index('lag_frames').loc[lag,'n_pairs']
            nn=pd.read_csv(source/'tables'/f'{video}_{pop}_nn.csv')
            assert nn.cdf.between(0,1).all() and np.all(np.diff(nn.cdf)>=0)
            a=pd.read_csv(source/'tables'/f'{video}_{pop}_vacf.csv')
            assert np.isclose(a.iloc[0].connected_vacf,1)
            assert a.connected_vacf.between(-1.000001,1.000001).all()
            checks.append(video+'/'+pop+': saved moments, survival, PDF, NN and velocity readback')
    target=base/f'report_{job}'
    target.mkdir(exist_ok=False)
    make_report(source,target)
    # Keep exact reviewed source in the final report directory.
    code=target/'source'; code.mkdir()
    for name in ['analysis.py','report.py','test_analysis.py','finalize_report.py','run.sh','finalize_report.sh']:
        shutil.copyfile(base/name,code/name)
    unit=json.loads((base/f'unit_tests_{job}.json').read_text())
    assert unit['passed'] and unit['tests_run']==16
    pdf=target/'male_crab_preliminary_report.pdf'
    validation={'unit_tests':unit,'input_output_checksums_passed':len(completion['artifacts']),
                'saved_estimator_readback_checks':checks,'actual_pdf_metadata_checked':False,
                'actual_pdf_pages_rendered':False}
    # Optional system PDF readers; no installation or environment changes.
    if shutil.which('pdfinfo'):
        p=subprocess.run(['pdfinfo',str(pdf)],check=True,capture_output=True,text=True)
        (target/'pdfinfo.txt').write_text(p.stdout)
        validation['actual_pdf_metadata_checked']=True
        lines=[x for x in p.stdout.splitlines() if x.startswith('Pages:')]
        assert lines and int(lines[0].split(':')[1])==17
    if shutil.which('pdftoppm'):
        renders=target/'pdf_rendered'; renders.mkdir()
        subprocess.run(['pdftoppm','-r','90','-png',str(pdf),str(renders/'page')],check=True,
                       stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        pngs=list(renders.glob('page-*.png')); assert len(pngs)==17
        validation['actual_pdf_pages_rendered']=True
    dump(target/'validation.json',validation)
    dump(target/'provenance.json',{'analysis_job_id':'15600144','report_job_id':job,
                                  'analysis_completion_sha256':sha(source/'COMPLETE.json'),
                                  'analysis_run':str(source.relative_to(base)),
                                  'scope':'Saved-result validation and editorial report; no new real-data estimators'})
    artifacts=[{'path':str(p.relative_to(target)),'bytes':p.stat().st_size,'sha256':sha(p)}
               for p in sorted(target.rglob('*')) if p.is_file()]
    dump(target/'COMPLETE.json',{'status':'complete','job_id':job,'artifacts':artifacts})
    print('REVIEWED_REPORT_COMPLETE',target,flush=True)


if __name__=='__main__': main()
