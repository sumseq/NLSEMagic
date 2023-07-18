function m = rmc_bisect(F,a,b,tol,par)

if(double(tol)<1e-15)
    dg = double(-log10(tol));
    digits(dg);
    a = vpa(a);
    b = vpa(b);
end

maxiters = 1000;
n = 0;
while(double(b-a) > double(tol))
    
    m = (a + (b-a)/2);
    
    Fa = real(feval(F,a,par));
    Fm = real(feval(F,m,par));
    
    if(sign(double(Fa)) == sign(double(Fm)))
        a = m;
    else
        b = m;
    end
    
    n = n+1;    
    if(n==maxiters)
        disp('Max iterations in bisection reached!');
        break
    end        
end
    
return