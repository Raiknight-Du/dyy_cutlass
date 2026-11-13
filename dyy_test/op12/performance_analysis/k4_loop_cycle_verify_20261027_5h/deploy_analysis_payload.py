from pathlib import Path,PurePosixPath
import tarfile,json,hashlib,sys,subprocess
R=Path(__file__).resolve().parent
assert R.name=='k4_loop_cycle_verify_20261027_5h' and R.parent.name=='performance_analysis'
receipt=json.loads((R/'LOCAL_ANALYSIS_TRANSPORT.json').read_text())
p=R/receipt['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['sha256']
with tarfile.open(p) as tf:
 for m in tf.getmembers():
  pp=PurePosixPath(m.name)
  assert not pp.is_absolute() and '..' not in pp.parts and m.isfile(),m.name
 tf.extractall(R)
print('ANALYSIS_PAYLOAD_HASH_AND_PATH_CHECK_PASS',flush=True)
subprocess.run([sys.executable,str(R/'seal_remote_evidence.py')],check=True)
