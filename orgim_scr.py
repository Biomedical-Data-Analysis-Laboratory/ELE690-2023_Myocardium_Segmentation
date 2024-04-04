import os
import pickle
import numpy as np
import pydicom
import organizeimage_TE as orgin

""" fr = "Ipython"
try:
    # Works with ipython
    #ip = get_ipython()
    get_ipython().magic("load_ext autoreload")
    get_ipython().magic("autoreload 2")
except:
    # If cpython interpreter is used
    fr = "cpython" """


def userGetOrg(study='haglag'):
    usersetfile = os.getcwd() + f"/{study}_set_orgin.p"
    if os.path.isfile(usersetfile):
        with open(usersetfile, "rb") as fp:
            usersettings = pickle.load(fp)
    else:
        print(usersetfile)
        raise FileNotFoundError
    return usersettings

def ptidx(prm,p):
    idx = [i for i in range(len(prm['PDLink'])) if prm['Pt'][p] == prm['PDLink'][i]]
    return idx

def maxval(prm, k, vrb=False):
    if prm['stud'] == 'haglag2':
        drecs = prm['drecs'][k]
    elif prm['stud'] == 'PM':
        drecs = np.ndarray.tolist(prm['drecs'][k])
    dcmf = os.path.join(prm['filepath'],drecs).replace('\\','/')
    #print(dcmf)
    ds = pydicom.dcmread(dcmf)
    img = pydicom.read_file(dcmf)
    r, c = img.pixel_array.shape
    imx = np.max(img.pixel_array)
    #print(ds.BitsStored)
    mv = 2**ds.BitsStored-1
    ba = ds.BitsAllocated
    bs = ds.BitsStored
    hb = ds.HighBit
    
    if vrb:
        print(f'Size {r}x{c} SamplesperPixel {ds.SamplesPerPixel}, Max pixel value {imx}')
        
    #print(2**ds.BitsStored-1)

    return r, c, bs, hb, imx, img

def print_prm(prm):
    keys = [*prm.keys()]
    for k in range(0,len(keys)):
        print('')
        #print(keys[k])
        s = 'prm[' + '\''  + keys[k] + '\'' + ']'
        if type(prm[keys[k]]) is int:
            s = s + ' = ' + str(prm[keys[k]])
        elif type(prm[keys[k]]) is bool:
            s = s + ' = ' + str(prm[keys[k]])
        elif type(prm[keys[k]]) is dict:
            s = s + ' = ' + str(prm[keys[k]])
        elif type(prm[keys[k]]) is str:
            s = s + ' = ' + prm[keys[k]]
        elif type(prm[keys[k]]) is list:
            s = s + ' = ' + str(prm[keys[k]][0:10]) + '(' + str(len(prm[keys[k]])) + ' elements)'
        elif type(prm[keys[k]]) is np.ndarray:
            s = s + ' = ' + str(prm[keys[k]][0:10]) + '(' + str(len(prm[keys[k]])) + ' elements)'
        else:
            s = s + ' = ' + str(type(prm[keys[k]]))
        print(s)

def main(study='haglag', PtID='AEA063', prm=None):
    if not prm:
        prm = userGetOrg(study)
        prm['filepath'] = '/home/prosjekt5/EKG/data/wmri/'
        if study =='haglag':
            prm['filepathDel'] = '/home/prosjekt5/EKG/data/wmri/konsensus_leik_stein/'
        elif study == 'vxvy':
            prm['filepathDel'] = '/home/prosjekt5/EKG/data/wmri/erlend/'
        elif study == 'PM':
            prm['filepath'] = '/home/prosjekt5/EKG/data/mri/PM/'
            prm['filepathDel'] = '/home/prosjekt5/EKG/data/mri/PM/erlend/'
    inD, b = orgin.organizeimage_TE(prm['filepath'],
                                    prm['filepathDel'],
                                    PtID,
                                    prm)
    Pt = [nm for nm in prm['Ptsgm'] if nm!='']
    return inD, b, prm, Pt
        
def main_old(study='haglag', PtID='AEA063'):
    prm = userGetOrg(study)
    prm['filepath'] = '/home/prosjekt5/EKG/data/wmri/'
    if study =='haglag':
        prm['filepathDel'] = '/home/prosjekt5/EKG/data/wmri/konsensus_leik_stein/'
    elif study == 'vxvy':
        prm['filepathDel'] = '/home/prosjekt5/EKG/data/wmri/erlend/'
    inD, b = orgin.organizeimage_TE(prm['filepath'],
                                    prm['filepathDel'],
                                    PtID,
                                    prm)
    Pt = [nm for nm in prm['Ptsgm'] if nm!='']
    return inD, b, prm, Pt


if __name__ == '__main__':
    inD, b, prm, Pt = main()    
