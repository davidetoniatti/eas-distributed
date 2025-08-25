using Random, LinearAlgebra, StatsBase
using BenchmarkTools
using Distributed, DistributedArrays
using MKL

@everywhere function _topk_indices!(top_idxs::Vector{Int}, top_vals::Vector{T}, v::AbstractVector{T}, k::Int) where T
    # Initialize the value buffer with the smallest possible value for type T.
    fill!(top_vals, typemin(T))
    # Initialize the index buffer with a placeholder (0 is a safe choice if indices are 1-based).
    fill!(top_idxs, 0)

    # These variables track the minimum value currently in our top-k buffer and its position.
    # This avoids having to search for the minimum every single iteration of the main loop.
    min_val_in_topk = typemin(T)
    min_pos_in_topk = 1

    # Iterate through each element of the input vector `v`.
    @inbounds for j in eachindex(v)
        val = v[j]

        # Check if the current value is larger than the smallest value in our top-k set.
        if val > min_val_in_topk
            # If it is larger, replace the smallest value in our buffer with this new value.
            top_idxs[min_pos_in_topk] = j
            top_vals[min_pos_in_topk] = val

            # We must find the new minimum for the next iteration's comparison.
            # We start the search by assuming the new minimum is the largest possible value.
            min_val_in_topk = typemax(T)

            # Perform a quick linear scan over the small `k`-sized buffer to find the new minimum.
            for i in 1:k
                if top_vals[i] < min_val_in_topk
                    min_val_in_topk = top_vals[i]
                    min_pos_in_topk = i
                end
            end
        end
    end

    return top_idxs
end

@everywhere function train_eas_classifier_optimized(X, y, P, k)
    d, n  = size(X)
    m = size(P,1)
    w = zeros(eltype(y),m)
    ct = zeros(Int64,m)
    
    # Determine the computation type based on the element types of X and M.
    T = promote_type(eltype(X), eltype(P))

    x_proj = Vector{Float64}(undef, m)
    top_idxs = Vector{Int}(undef, k)
    top_vals = Vector{T}(undef, k)
    
    for i in 1:n
        x_view = X[:, i]
        mul!(x_proj, P, x_view)
        _topk_indices!(top_idxs, top_vals, x_proj, k)

        label = y[i]
        for j in top_idxs
            w[j] += label
            ct[j] += 1
        end
    end
    
    return w, ct
end