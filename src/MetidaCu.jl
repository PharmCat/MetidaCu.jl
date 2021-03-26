module MetidaCu
    using LinearAlgebra, CUDA
    import MetidaNLopt, Metida
    import MetidaNLopt: reml_sweep_β_cuda#, cudata
    import Metida: LMM, AbstractLMMDataBlocks, rmat_base_inc!, zgz_base_inc!, logerror!

    struct LMMDataBlocks{T1, T2} <: AbstractLMMDataBlocks
        # Fixed effect matrix views
        xv::T1
        # Responce vector views
        yv::T2
        function LMMDataBlocks(xv::Matrix{T}, yv::Vector{T}, vcovblock::Vector) where T
            x = Vector{CuArray{T, 2}}(undef, length(vcovblock))
            y = Vector{CuArray{T, 1}}(undef, length(vcovblock))
            for i = 1:length(vcovblock)
                x[i] = CuArray(view(xv, vcovblock[i],:))
                y[i] = CuArray(view(yv, vcovblock[i]))
            end
            new{typeof(x), typeof(y)}(x, y)
        end
        function LMMDataBlocks(lmm)
            return LMMDataBlocks(lmm.data.xv, lmm.data.yv, lmm.covstr.vcovblock)
        end
    end

    function MetidaNLopt.cudata(lmm::LMM)
        LMMDataBlocks(lmm.data.xv, lmm.data.yv, lmm.covstr.vcovblock)
    end

    function MetidaNLopt.reml_sweep_β_cuda(lmm::LMM, θ::Vector{T}) where T
        reml_sweep_β_cuda(lmm,  MetidaNLopt.cudata(lmm), θ)
    end

    function MetidaNLopt.reml_sweep_β_cuda(lmm::LMM, data::AbstractLMMDataBlocks, θ::Vector{T}) where T
        n             = length(lmm.covstr.vcovblock)
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
        logdetθ₂      = zero(T)
        try
            @inbounds @simd for i = 1:n
                q    = length(lmm.covstr.vcovblock[i])
                qswm = q + lmm.rankx
                V    = zeros(T, q, q)
                Metida.zgz_base_inc!(V, θ, lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
                Metida.rmat_base_inc!(V, θ[lmm.covstr.tr[end]], lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
                A[i] = CuArray(V)
            #-------------------------------------------------------------------
            # Cholesky
                A[i] = LinearAlgebra.LAPACK.potrf!('U', A[i])[1]
                θ₁  += sum(log.(diag(A[i])))*2
                vX   = LinearAlgebra.LAPACK.potrs!('U', A[i], copy(data.xv[i]))
                vy   = LinearAlgebra.LAPACK.potrs!('U', A[i], copy(data.yv[i]))
                CUDA.CUBLAS.gemm!('T', 'N', one(T), data.xv[i], vX, one(T), θ₂tc)
                CUDA.CUBLAS.gemv!('T', one(T), data.xv[i], vy, one(T), βtc)
            #-------------------------------------------------------------------
            end
        # Beta calculation
            copyto!(θ₂, θ₂tc)
            LinearAlgebra.LAPACK.potrf!('U', θ₂tc)
            copyto!(β, LinearAlgebra.LAPACK.potrs!('U', θ₂tc, βtc))
        # θ₃ calculation
            @inbounds @simd for i = 1:n
                r = CUDA.CUBLAS.gemv!('N', -one(T), data.xv[i], βtc, one(T), copy(data.yv[i]))
                vr   = LinearAlgebra.LAPACK.potrs!('U', A[i], copy(r))
                θ₃  += r'*vr
            end
            logdetθ₂ = logdet(θ₂)
        catch e
            logerror!(e, lmm)
            return (Inf, nothing, nothing, nothing, false)
        end
        return   θ₁ + logdetθ₂ + θ₃ + c, β, θ₂, θ₃, true
    end
end # module
