module EaSDistributed

using Random, LinearAlgebra, .Threads, StatsBase
using Distributed, DistributedArrays

"""
    rupm(m::Int, d::Int, seed::Int) ->
    Matrix{Float64}

Creates a random projection matrix of size `m x d`, whose row are sampled
i.i.d. from the uniform distribution over S^{d-1}.

# Arguments
- `m::Int`: Number of rows.
- `d::Int`: Number of columns.
- `seed::Int`: Seed for initializing the random number generators.

# Returns
- `Matrix{Float64}`: The generated projection matrix.
"""
function rupm(m::Int, d::Int; seed::Int=42)
    rng = MersenneTwister(seed)
    mat = randn(rng, m, d)

    # Normalize each row in-place to have unit L2 norm.
    @inbounds for i in 1:m
        # Get a view of the current row to avoid copying data.
        row_view = @view mat[i, :]

        # Calculate the norm of the row.
        row_norm = norm(row_view)

        # Normalize the row. Add a small epsilon for numerical stability.
        # This prevents division by zero if a row happens to be all zeros.
        mat[i, :] ./= (row_norm + eps(eltype(mat)))
    end

    return mat
end


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


@everywhere function train_eas_classifier(X, y, P, k)
    n = size(X, 2)
    m = size(P, 1)
    w = zeros(eltype(y), m)
    ct = zeros(Int64, m)

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

function distributed_train_eas_classifier(X::DArray, y::DArray, P, k::Int)
    # Each worker compute w, ct on localpart(X,y)
    futures = [@spawnat p train_eas_classifier_optimized(localpart(X), localpart(y), P, k) for p in workers()]

    # Fetch results
    results = fetch.(futures)

    # Reduce operation
    m = size(P,1)
    w_total, ct_total = zeros(eltype(y),m), zeros(Int64,m)
    for (w, ct) in results
        w_total .+= w
        ct_total .+= ct
    end

    w_normalized = zeros(Float64, length(w_total))
    valid_indices = ct_total .> 0
    
    w_normalized[valid_indices] .= w_total[valid_indices] ./ ct_total[valid_indices]

    return w_normalized, ct_total
end


function infer_eas_classifier(X, k, w, P)
    n = size(X, 2)
    m = length(w)
    scores = zeros(n)

    T = promote_type(eltype(X), eltype(P))

    x_proj = Vector{Float64}(undef, m)
    top_idxs = Vector{Int}(undef, k)
    top_vals = Vector{T}(undef, k)
    
    for i in 1:n
        x_view = X[:, i]
        mul!(x_proj, P, x_view)
        _topk_indices!(top_idxs, top_vals, x_proj, k)

        scores[i] = sum(w[top_idxs]) / k
    end

    y_pred = Int.(scores .>= 0.5)
    return y_pred
end

end # module