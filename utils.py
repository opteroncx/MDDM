'''
这里存放一些工具代码
'''
import datetime
import os
import shutil

def save_experiment():
    root_path = './experiments'
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    code_path = os.path.join(root_path,t)
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    copy_files('./',code_path)
    print('code copied to ',code_path)

def copy_files(source, target):
    files = os.listdir(source)
    for f in files:
        if f[-3:] == '.py' or f[-3:] == '.sh':
            print(f)
            shutil.copy(source+f, target)

    

if __name__ == "__main__":
    save_experiment()
