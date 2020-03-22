function CRC.rrule(::typeof(logpdf), d::Exponential, x)
    v = logpdf(d, x)
    function pullback(y)
        g = @thunk(gradlogpdf(d, x))
        dθ = @thunk(
            if x < 0
                zero(x)
            else
                -inv(d.θ) + x * inv(d.θ^2)
            end
        )
        return (NO_FIELDS, g, dθ)
    end
    return (v, pullback)
end

function CRC.rrule(::typeof(logpdf), d::Normal, x)
    v = logpdf(d, x)
    function pullback(y)
        μ = mean(d)
        σ = std(d)
        g = @thunk(gradlogpdf(d, x))
        dμ = @thunk((x - μ) / σ^2)
        dσ = @thunk(-inv(σ) + abs2(x - μ) * σ^(-3))
        return (NO_FIELDS, g, dμ, dσ)
    end
    return (v, pullback)
end

function CRC.rrule(::typeof(logpdf), d::Arcsine, x)
    v = logpdf(d, x)
    function pullback(y)
        T = promote_type(typeof(x), partype(d))
        (a, b) = support(d)
        g = @thunk(gradlogpdf(d, x))
        da = if x > b || x < a
                zero(T)
            else
                inv(2 * (x - a))
            end
        
        db = if x > b || x < a
                zero(T)
            else
                -inv(2 * (b - x))
            end
        return (NO_FIELDS, g, da, db)
    end
    return (v, pullback)
end

function CRC.rrule(::typeof(logpdf), d::Beta, x)
    v = logpdf(d, x)
    function pullback(y)
        T = promote_type(typeof(x), partype(d))
        (α, β) = params(d)
        g = @thunk(gradlogpdf(d, x))
        dα = if !(0 ≤ x ≤ 1)
                zero(T)
            else
                log(x) - digamma(α) + digamma(α + β)
            end
        dβ = if !(0 ≤ x ≤ 1)
                zero(T)
            else
                log(1 - x) - digamma(β) + digamma(α + β)
            end
        return (NO_FIELDS, g, dα, dβ)
    end
    return (v, pullback)
end
