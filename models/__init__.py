from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.BEVANet
import models.LSKA
import models.PPM
import models.conv
import models.FB

module_dict_LKA = {
    "LSKA": models.LSKA.LSKA,
    "SDLSKA": models.LSKA.SDLSKA,
    "SLAK": models.LSKA.SLAK
}

module_dict_conv = {
    "DWConv": models.conv.DWConv,
    "PDWConv": models.conv.PDWConv,
    "PWConv": models.conv.PWConv,
    "GSPWConv": models.conv.GSPWConv,
    "PGSPWConv": models.conv.PGSPWConv,
}

module_dict_PPM = {
    "DAPPM": models.PPM.DAPPM,
    "PAPPM": models.PPM.PAPPM,
    "DLKPPM": models.PPM.DLKPPM,
    "PLKPPM": models.PPM.PLKPPM,
}
module_dict_FB = {
    "Bag": models.FB.Bag,
    "Light_Bag": models.FB.Light_Bag,
    "BGAF": models.FB.BGAF,
}

module_dict = {
    "LKA": module_dict_LKA,
    "conv": module_dict_conv,
    "PPM": module_dict_PPM,
    "FB": module_dict_FB,
}