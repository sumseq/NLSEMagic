function F_OM_str_res = F_OM_str(OM_str,OM)


T = atanh(sqrt(3./(16.*OM_str)) - sqrt(3./(16.*OM_str) - 1));

%T = sqrt(3./(16.*OM_str)) - sqrt(3./(16.*OM_str) - 1);
F_OM_str_res = OM_str./2 + 3./32 - (1./(4.*T)).*sqrt((3.*OM_str)./16) - OM;

%if(isreal(F_OM_str_res)==0)
 %   F_OM_str_res = -0.0001;
%end

return