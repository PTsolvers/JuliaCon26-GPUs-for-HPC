# Shared helpers for the workshop scripts.
#
# Infrastructure only -- no physics. Each script keeps its own `lap`, its own
# kernels and its own `check`, so that any one of them can be read on its own:
# the plain-vs-KA diff is the teaching point and only works if both files are
# self-contained.
using Printf, CairoMakie
# NB no `using KernelAbstractions` here: `backend_info` and `bench` reference it, but
# only the KA scripts call them and they load it themselves. Keeping it out lets the
# plain-Julia script include this file for the figure helpers without pulling in KA.

# output/<script name>/ at the repo root, created on demand.
# Call as `outdir(@__FILE__)`: @__FILE__ expands at the call site, so the folder
# follows the calling script's name without hardcoding it.
function outdir(file::AbstractString)
    name = splitext(basename(file))[1]
    isempty(name) && (name = "unnamed")      # e.g. when eval'd without a file
    d = normpath(joinpath(@__DIR__, "..", "output", name))
    mkpath(d)
    return d
end

# KA's CPU backend threads over workgroups, so the thread count matters there
backend_info(backend) = backend isa CPU ? "CPU ($(Threads.nthreads()) threads)" :
                                          string(nameof(typeof(backend)))

# effective memory throughput: n_arrays passes over the grid, cache ignored
Teff(narr, nx, ny, t, T) = narr * nx * ny * sizeof(T) / t / 1e9

# best-of-ntrial mean over nrep launches, so scheduling hiccups drop out.
# NB this is a *burst* measurement -- a real time loop runs ~10% slower.
function bench(backend, k, args; nrep=50, ntrial=5)
    k(args...)                                  # warm up / compile
    KernelAbstractions.synchronize(backend)
    best = Inf
    for _ in 1:ntrial
        KernelAbstractions.synchronize(backend); t0 = time()
        for _ in 1:nrep
            k(args...)
        end
        KernelAbstractions.synchronize(backend)
        best = min(best, (time() - t0) / nrep)
    end
    return best
end

# Single-field figure: heatmap of C_v with a colorbar. Returns (fig, ax, plt).
function field_figure(C_v; title="C", colormap=:turbo, colorrange=nothing)
    nx, ny = size(C_v)
    fig = Figure(; size=(600, 500))
    ax  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title)
    plt = colorrange === nothing ? heatmap!(ax, 1:nx, 1:ny, C_v; colormap) :
                                   heatmap!(ax, 1:nx, 1:ny, C_v; colormap, colorrange)
    Colorbar(fig[1, 1][1, 2], plt)
    return fig, ax, plt
end

# Cahn-Hilliard figure: heatmap of C, with the free energy as an inset bottom-right.
# Returns (fig, axs, plt, vid); drive it from the time loop with
#     axs[1].title = ...        plt[1][3] = C_v        plt[2][1] = Fs
# `bg` is an opaque panel behind the inset: an axis draws its title and tick labels
# OUTSIDE its own box, so without it they land straight on the heatmap. The two
# translate! calls lift panel and inset above the heatmap (z = 99 / 100) -- without
# them both render underneath it and disappear.
# px_per_unit=1 keeps the gif at the figure's own pixel size. Makie defaults to 2
# (retina), which is 4x the pixels and made a 40-frame gif ~24 MB instead of ~6 MB.
# TRAP: any `save(path, fig)` interleaved with `recordframe!` must use the SAME
# px_per_unit, or Cairo segfaults. Hence `savepng` below -- use it for frames.
function ch_figure(C_v, Fs, tmax, Fmax; colormap=:balance, colorrange=(-1, 1), px_per_unit=1)
    nx, ny = size(C_v)
    fig = Figure(; size=(620, 560))
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="C"),
           Axis(fig[1, 1][1, 1]; width=Relative(0.31), height=Relative(0.165),
                halign=0.945, valign=0.115, limits=(0, tmax, 0, Fmax),
                backgroundcolor=:white, title="free energy F", titlesize=9,
                xticklabelsize=7, yticklabelsize=7, xticksize=2, yticksize=2,
                xticks=[0, round(tmax)], yticks=[0, round(Fmax, sigdigits=2)],
                xlabelvisible=false, ylabelvisible=false,
                xgridvisible=false, ygridvisible=false))
    bg  = Box(fig[1, 1][1, 1]; width=Relative(0.43), height=Relative(0.28),
              halign=0.98, valign=0.03, color=:white,
              strokecolor=(:gray40, 0.7), strokewidth=1)
    plt = (heatmap!(axs[1], 1:nx, 1:ny, C_v; colormap, colorrange),
           lines!(axs[2], Fs; color=:crimson, linewidth=2))
    Colorbar(fig[1, 1][1, 2], plt[1])
    translate!(bg.blockscene,     0, 0, 99)
    translate!(axs[2].blockscene, 0, 0, 100)
    return fig, axs, plt, VideoStream(fig; px_per_unit)
end

# Save a figure frame at the same px_per_unit as ch_figure's VideoStream (see above).
savepng(path, fig; px_per_unit=1) = save(path, fig; px_per_unit)

# Save a VideoStream as a gif at the requested framerate.
# TRAP: the `framerate` given to VideoStream(fig; framerate=…) is IGNORED for gif
# output -- only the one passed to `save` takes effect, and its default is 25 fps.
# Always go through this helper.
savegif(path, vid; framerate=5) = save(path, vid; framerate)
