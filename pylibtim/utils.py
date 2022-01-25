from enum import Enum

class FlatSE_Enum(Enum):
	make2DN4  = 1
	make2DN8  = 2
	make3DN6  = 3
	make3DN18 = 4
	make3DN26 = 5

def FlatSE(se):
    return FlatSE_Enum[se].value
