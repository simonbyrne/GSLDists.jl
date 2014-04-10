module GSLDists

using GSL
using Distributions

import Distributions: pdf, cdf, ccdf, quantile, cquantile

export GSLDist


immutable GSLDist{T<:UnivariateDistribution,S<:ValueSupport} <: UnivariateDistribution{S}
    dist::T
end
GSLDist{S<:ValueSupport}(d::UnivariateDistribution{S}) = GSLDist{typeof(d),S}(d)


macro gsl_dist(T, b)
    pdf_fn = symbol(string("ran_",b,"_pdf"))
    cdf_fn = symbol(string("cdf_",b,"_P"))
    ccdf_fn = symbol(string("cdf_",b,"_Q"))
    quantile_fn = symbol(string("cdf_",b,"_Pinv"))
    cquantile_fn = symbol(string("cdf_",b,"_Qinv"))

    Ty = eval(T)
    pn = Ty.names                       # parameter names

    x_in = :x
    if Ty <: DiscreteDistribution
        x_in = :(int(x))
    end


    if length(pn) == 1
        
        p1  = Expr(:quote, pn[1])
        quote
            global pdf, cdf, ccdf, quantile, cquantile

            pdf{S}(d::GSLDist{$T,S}, x::Real) = ($pdf_fn)($x_in, d.dist.($p1))
            cdf{S}(d::GSLDist{$T,S}, x::Real) = ($cdf_fn)($x_in, d.dist.($p1))
            ccdf{S}(d::GSLDist{$T,S}, x::Real) = ($ccdf_fn)($x_in, d.dist.($p1))
            quantile{S}(d::GSLDist{$T,S}, p::Real) = ($quantile_fn)(p, d.dist.($p1))
            cquantile{S}(d::GSLDist{$T,S}, p::Real) = ($cquantile_fn)(p, d.dist.($p1))
        end

    elseif length(pn) == 2
    
        p1 = Expr(:quote, pn[1])
        p2 = Expr(:quote, pn[2])

        quote
            global pdf, cdf, ccdf, quantile, cquantile

            pdf{S}(d::GSLDist{$T,S}, x::Real) = ($pdf_fn)($x_in, d.dist.($p1), d.dist.($p2))
            cdf{S}(d::GSLDist{$T,S}, x::Real) = ($cdf_fn)($x_in, d.dist.($p1), d.dist.($p2))
            ccdf{S}(d::GSLDist{$T,S}, x::Real) = ($ccdf_fn)($x_in, d.dist.($p1), d.dist.($p2))
            quantile{S}(d::GSLDist{$T,S}, p::Real) = ($quantile_fn)(p, d.dist.($p1), d.dist.($p2))
            cquantile{S}(d::GSLDist{$T,S}, p::Real) = ($cquantile_fn)(p, d.dist.($p1), d.dist.($p2))
        end
    end
end

# @gsl_dist Cauchy cauchy #20.8

@gsl_dist Gamma gamma         #20.14
@gsl_dist Uniform flat        #20.15
@gsl_dist LogNormal lognormal #20.16
@gsl_dist Chisq chisq         #20.17
@gsl_dist FDist fdist         #20.18
# @gsl_dist TDist tdist         #20.19
@gsl_dist Beta beta           #20.20
@gsl_dist Logistic logistic   #20.21
# @gsl_dist Pareto pareto       #20.22 wrong order
@gsl_dist Weibull weibull     #20.24

@gsl_dist Poisson poisson     #20.29

end # module
