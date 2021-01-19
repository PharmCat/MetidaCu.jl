module MetidaCu
    using LinearAlgebra
    using Metida, MetidaNLopt, CUDA
    import Metida: LMM, gmat_base_z2!, rmat_basep_z2!

    function MetidaNLopt.reml_sweep_β_cuda(lmm::LMM, θ::Vector{T}) where T
        n             = length(lmm.data.block)
        N             = length(lmm.data.yv)
        c             = (N - lmm.rankx)*log(2π)
        #-----------------------------------------------------------------------
        θ₁            = zero(T)
        θ₂            = zeros(T, lmm.rankx, lmm.rankx)
        θ₂tc          = CUDA.zeros(T, lmm.rankx, lmm.rankx)
        θ₃            = zero(T)
        βtc           = CUDA.zeros(T, lmm.rankx)
        β             = Vector{T}(undef, lmm.rankx)
        A             = Vector{CuArray{T, 2}}(undef, n)
        X             = Vector{CuArray{T, 2}}(undef, n)
        y             = Vector{CuArray{T, 1}}(undef, n)
        q             = zero(Int)
        qswm          = zero(Int)
        logdetθ₂      = zero(T)
        @inbounds for i = 1:n
            q    = length(lmm.data.block[i])
            qswm = q + lmm.rankx
            V    = zeros(T, q, q)
            Metida.gmat_base_z2!(V, θ, lmm.covstr, lmm.data.block[i], lmm.covstr.sblock[i])
            Metida.rmat_basep_z2!(V, θ[lmm.covstr.tr[end]], lmm.covstr, lmm.data.block[i], lmm.covstr.sblock[i])
            A[i] = CuArray(V)
            X[i] = CuArray(view(lmm.data.xv,  lmm.data.block[i], :))
            y[i] = CuArray(view(lmm.data.yv, lmm.data.block[i]))
            #-------------------------------------------------------------------
            #Cholesky
            A[i] = LinearAlgebra.LAPACK.potrf!('L', A[i])[1]
            θ₁  += logdet(Cholesky(Matrix(A[i]), 'L', 0))
            vX   = LinearAlgebra.LAPACK.potrs!('L', A[i], copy(X[i]))
            vy   = LinearAlgebra.LAPACK.potrs!('L', A[i], copy(y[i]))
            CUDA.CUBLAS.gemm!('T', 'N', one(T), X[i], vX, one(T), θ₂tc)
            CUDA.CUBLAS.gemv!('T', one(T), X[i], vy, one(T), βtc)
            #-------------------------------------------------------------------
        end
        #Beta calculation
        copyto!(θ₂, θ₂tc)
        LinearAlgebra.LAPACK.potrf!('L', θ₂tc)
        copyto!(β, LinearAlgebra.LAPACK.potrs!('L', θ₂tc, βtc))
        # θ₃ calculation
        @simd for i = 1:n
            r = CUDA.CUBLAS.gemv!('N', -one(T), X[i], βtc,
            one(T), y[i])
            vr   = LinearAlgebra.LAPACK.potrs!('L', A[i], copy(r))
            θ₃  += r'*vr
        end
        logdetθ₂ = logdet(θ₂)
        return   θ₁ + logdetθ₂ + θ₃ + c, β, θ₂, θ₃
    end
end # module
