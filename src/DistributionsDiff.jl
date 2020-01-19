module DistributionsDiff

using Distributions
import ChainRulesCore
const CRC = ChainRulesCore
using ChainRulesCore: NO_FIELDS, @thunk

include("univariates.jl")

end # module
