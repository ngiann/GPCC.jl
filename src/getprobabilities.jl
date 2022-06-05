function getprobabilities(out)

    Q = reduce(hcat, out)'

    nfolds = size(Q, 2)

    aux = vec(mean(reduce(hcat, [exp.(Q[:,i] .- logsumexp(Q[:,i])) for i in 1:nfolds]), dims=2))

    reshape(aux, size(out))

end

# function getprobabilities(cvresults)
#
#     aux = mean.(cvresults)
#
#     pr = exp.(aux .- logsumexp(aux))
#
#     pr / sum(pr) # sometimes pr will not sum up to 1 exactly, this line helps
#
# end
