
function informuser(; seed = seed, iterations = iterations, numberofrestarts = numberofrestarts,
                    JITTER = JITTER, ρmin = ρmin, Σb = Σb)

    colourprint(@sprintf("Running gpcc2vi with random seed %d\n", seed), foreground = :light_blue, bold = true)
    @printf("\t iterations             = %d\n", iterations)
    @printf("\t numberofrestarts       = %d\n", numberofrestarts)
    @printf("\t JITTER                 = %e\n", JITTER)
    @printf("\t ρmin                   = %f\n", ρmin)
    @printf("\t ρmax                   = %f\n", ρmax)
    @printf("\t Σb                     = "); map(x->@printf("%.3f ",x), diag(Σb)); @printf("\n")
end


#---------------------------------------------------

rbf(xᵢ,xⱼ ; ℓ²=1.0) = exp(-0.5*(xᵢ-xⱼ)^2/(2ℓ²))

#---------------------------------------------------

function Qmatrix(Narray)

    L = length(Narray) # number of time series

    local Q = zeros(sum(Narray), L)

    for l in 1:L

        Q[1+sum(Narray[1:l-1]):sum(Narray[1:l]) ,l] .= 1.0

    end

    Q

end
