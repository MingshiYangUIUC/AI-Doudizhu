import subprocess
import os
import glob
import shutil

os.chdir(os.path.dirname(__file__))

subprocess.run(f"python \"{os.path.join(os.path.dirname(__file__),'setup.py')}\" build_ext --inplace",shell=True,check=False)

dest_dir = os.path.join(os.path.dirname(__file__),'..')

for file in glob.glob('*.pyd'):
    print(file)
    try:
        os.remove(os.path.join(dest_dir,file))
    except:
        pass
    shutil.move(file, dest_dir)

for file in glob.glob('*.so'):
    print(file)
    try:
        os.remove(os.path.join(dest_dir,file))
    except:
        pass
    shutil.move(file, dest_dir)