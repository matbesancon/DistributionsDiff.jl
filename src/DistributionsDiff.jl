module DistributionsDiff

using Distributions
import ChainRulesCore
const CRC = ChainRulesCore
using ChainRulesCore: NO_FIELDS, @thunk

using SpecialFunctions: loggamma, digamma

include("univariates.jl")
include("logpdf.jl")
end # module
