function F_OM_va = F_OM_va(OM_str)

T = atanh(sqrt(3./(16.*OM_str)) - sqrt(3./(16.*OM_str) - 1));

F_OM_va =  OM_str./2 + 3./32 - (1./(4.*T)).*sqrt((3.*OM_str)./16);

return