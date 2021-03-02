module MetidaCu
    using LinearAlgebra, CUDA
    import MetidaNLopt, Metida
    import MetidaNLopt: reml_sweep_β_cuda#, cudata
    import Metida: LMM, AbstractLMMDataBlocks, rmat_base_inc!, zgz_base_inc!
    #using TimerOutputs

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
        #to = TimerOutput()
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
        #X             = Vector{CuArray{T, 2}}(undef, n)
        #y             = Vector{CuArray{T, 1}}(undef, n)
        q             = zero(Int)
        qswm          = zero(Int)
        logdetθ₂      = zero(T)
        @inbounds for i = 1:n
            q    = length(lmm.covstr.vcovblock[i])
            qswm = q + lmm.rankx
            #@timeit to "V matrix" begin
            V    = zeros(T, q, q)
            Metida.zgz_base_inc!(V, θ, lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
            Metida.rmat_base_inc!(V, θ[lmm.covstr.tr[end]], lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
            #end
            #@timeit to "CuArray"
            A[i] = CuArray(V)
            #X[i] = CuArray(view(lmm.data.xv,  lmm.covstr.vcovblock[i], :))
            #y[i] = CuArray(view(lmm.data.yv, lmm.covstr.vcovblock[i]))
            #-------------------------------------------------------------------
            #Cholesky
            #@timeit to "potrf! 1"
            A[i] = LinearAlgebra.LAPACK.potrf!('L', A[i])[1]
            try
                #@timeit to "logdet 1" begin
                    θ₁  += sum(log.(diag(A[i])))*2
                #end
                #θ₁  += logdet(Cholesky(Matrix(A[i]), 'L', 0))
            catch
                lmmlog!(lmm, LMMLogMsg(:ERROR, "θ₁ not estimated during REML calculation, V isn't positive definite or |V| less zero."))
                return (1e100, nothing, nothing, 1e100)
            end
            #@timeit to "potrs! 1"
            vX   = LinearAlgebra.LAPACK.potrs!('L', A[i], copy(data.xv[i]))
            #@timeit to "potrs! 2"
            vy   = LinearAlgebra.LAPACK.potrs!('L', A[i], copy(data.yv[i]))
            #@timeit to "gemm! 1"
            CUDA.CUBLAS.gemm!('T', 'N', one(T), data.xv[i], vX, one(T), θ₂tc)
            #@timeit to "gemv! 1"
            CUDA.CUBLAS.gemv!('T', one(T), data.xv[i], vy, one(T), βtc)
            #-------------------------------------------------------------------
        end
        #Beta calculation
        #@timeit to "θ₂"
        copyto!(θ₂, θ₂tc)
        LinearAlgebra.LAPACK.potrf!('L', θ₂tc)
        #@timeit to "β"
        copyto!(β, LinearAlgebra.LAPACK.potrs!('L', θ₂tc, βtc))
        # θ₃ calculation
        @simd for i = 1:n
            #@timeit to "gemv! 2"
            r = CUDA.CUBLAS.gemv!('N', -one(T), data.xv[i], βtc, one(T), copy(data.yv[i]))
            #@timeit to "potrs! 3"
            vr   = LinearAlgebra.LAPACK.potrs!('L', A[i], copy(r))
            θ₃  += r'*vr
        end
        #@timeit to "logdet 2"
        logdetθ₂ = logdet(θ₂)
        #println(to)
        return   θ₁ + logdetθ₂ + θ₃ + c, β, θ₂, θ₃
    end
end # module
