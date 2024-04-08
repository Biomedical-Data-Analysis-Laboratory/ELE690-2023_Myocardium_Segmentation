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
            prm = pickle.load(fp)
        if type(prm['drecs'][0]) == np.ndarray:
            prm['drecs'] = [np.ndarray.tolist(d) for d in prm['drecs'] if type(d) == np.ndarray]
    else:
        print(usersetfile)
        raise FileNotFoundError
    return prm

def ptidx(prm,p):
    # Get drecs indexes for a given patient (p is name or number)
    if type(p) == int:
        idx = [i for i in range(len(prm['PDLink'])) if prm['Pt'][p] == prm['PDLink'][i]]
    elif type(p) == str:
        Pt = p
        idx = [i for i in range(len(prm['PDLink'])) if Pt == prm['PDLink'][i]]
    return idx

def maxval(prm, k, vrb=False):
    # Get pixel value information from kth drecs entry
    drecs = prm['drecs'][k]
    dcmf = os.path.join(prm['filepath'],drecs).replace('\\','/')
    #print(dcmf)
    ds = pydicom.dcmread(dcmf)
    img = pydicom.read_file(dcmf)
    
    r, c = img.pixel_array.shape
    imx = np.max(img.pixel_array)
    imx_n = imx*float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    #print(ds.BitsStored)
    mv = 2**ds.BitsStored-1
    ba = ds.BitsAllocated
    bs = ds.BitsStored
    hb = ds.HighBit
    
    if vrb:
        print(f'Size {r}x{c} SamplesperPixel {ds.SamplesPerPixel}, Max pixel value {imx}')
        
    #print(2**ds.BitsStored-1)

    return r, c, bs, hb, imx, imx_n, img 

def get_pixvalinfo(prm, ld=False, vrb=False):
    infofile = os.getcwd() + f'/{prm["stud"]}_valinfo.p'
    if os.path.isfile(infofile) and ld:
        with open(infofile, "rb") as fp:
            S = pickle.load(fp)
    else:
        S = {'Pt': [], 'min': [], 'max': [], 'slices': []}
        
        Np = len(prm['Pt'])
        for p in range(Np):
            if vrb:
                print(f'\n{p+1}:{len(prm["Pt"])} {prm["Pt"][p]}')
            #print(prm["Pt"][p])
            idx = ptidx(prm,p)
            files = [prm['drecs'][i].replace('\\','/') for i in idx]
            MV = [] # rows
            BA = [] # columns
            BS = []
            HB = []
            IM = []
            IM_n = []
            S2 = {}
            for k in idx:
                mv, ba, bs, hb, imx, imx_n, img = maxval(prm,k)
                MV.append(mv)
                BA.append(ba)
                BS.append(bs)
                HB.append(hb)
                IM.append(imx)
                IM_n.append(imx_n)
            rows = np.unique(MV)
            cols = np.unique(BA)
            bs = np.unique(BS)
            hb = np.unique(HB)
            im = np.unique(IM)
            im_n = np.unique(IM_n)
            S['Pt'].append(prm['Pt'][p])
            S2[prm['Pt'][p]] = {'file': files, 'max': IM, \
                                'max_n': IM_n, 'bits_stored': BS}            
            
            if len(im) > 0:
                S['min'].append(np.min(im))
                S['max'].append(np.max(im))
            else:
                S['min'].append(np.nan)
                S['max'].append(np.nan)
            S['slices'].append(S2)
            #print(S)
            if vrb:
                try:
                    print(f'   Size {rows}x{cols} Pixel range {np.min(im)}-{np.max(im)} (normalised {np.min(im_n):.0f}-{np.max(im_n):.0f})')
                except:
                    print('Not printable')

        with open(infofile, "wb") as fp:
            pickle.dump(S,fp)
            
    return S

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
