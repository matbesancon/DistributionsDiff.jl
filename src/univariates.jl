
## gradlogpdf ##

# TODO incorrect, NO_FIELDS is wrong
# function CRC.rrule(::typeof(logpdf), d::UnivariateDistribution, x)
    # v = logpdf(d, x)
    # pullback(y) = (NO_FIELDS, d. @thunk(gradlogpdf(d, x)))
    # return (v, pullback)
# end

function CRC.rrule(::Type{D}, θ) where {D <: Exponential}
    function D_pullback(dy)
        return (NO_FIELDS, dy.θ)
    end
    return D(θ), D_pullback
end

function CRC.rrule(::Type{D}, a, b) where {D <: Union{Uniform, Arcsine}}
    function D_pullback(dy)
        return (NO_FIELDS, dy.a, dy.b)
    end
    return D(a, b), D_pullback
end

function CRC.rrule(::Type{D}, α, β) where {D <: Union{Beta, BetaPrime}}
    function D_pullback(dy)
        return (NO_FIELDS, dy.α, dy.β)
    end
    return D(α, β), D_pullback
end

function CRC.rrule(::Type{D}, µ, σ) where {D <: Union{Normal, LogNormal}}
    function D_pullback(dy)
        return (NO_FIELDS, dy.µ, dy.σ)
    end
    return D(µ, σ), D_pullback
end

function CRC.rrule(::Type{<:Weibull}, α, θ)
    function weibull_pullback(dy)
        return (NO_FIELDS, dy.α, dy.θ)
    end
    return Weibull(α, θ), weibull_pullback
end
