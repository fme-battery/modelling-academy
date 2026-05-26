
###############################
# Plot dashboard

function plot_dashboard(output; plot_type="simple")

    ts = output.time_series

    t = ustrip.(ts["Time"] ./ u"hr")
    I = ts["Current"]
    E = ts["Voltage"]

    if plot_type == "simple"

        p_current = plot(t, I;
            linewidth=2, marker=:circle, label="Current",
            xlabel="Time / h", ylabel="Current / A",
            title="Current / A"
        )

        p_voltage = plot(t, E;
            linewidth=2, marker=:circle, label="Voltage",
            xlabel="Time / h", ylabel="Voltage / V",
            title="Voltage / V"
        )

        return plot(p_current, p_voltage;
            layout=(2,1), size=(1200, 800))

    elseif plot_type == "contour"

        states = output.states

        # --- extract data ---
        NeAm_pos = states["NegativeElectrode"]["ActiveMaterial"]["Position"]
        PeAm_pos = states["PositiveElectrode"]["ActiveMaterial"]["Position"]
        Elyte_pos = states["Electrolyte"]["Position"]

        NeAm_conc = states["NegativeElectrode"]["ActiveMaterial"]["SurfaceConcentration"]
        PeAm_conc = states["PositiveElectrode"]["ActiveMaterial"]["SurfaceConcentration"]
        Elyte_conc = states["Electrolyte"]["Concentration"]

        NeAm_pot = states["NegativeElectrode"]["ActiveMaterial"]["Potential"]
        PeAm_pot = states["PositiveElectrode"]["ActiveMaterial"]["Potential"]
        Elyte_pot = states["Electrolyte"]["Potential"]

        # --- top plots ---
        p_current = plot(t, I;
            linewidth=2, marker=:circle,
            ylabel="Current / A", xlabel="Time / h", title="Current / A")

        p_voltage = plot(t, E;
            linewidth=2, marker=:circle,
            ylabel="Voltage / V", xlabel="Time / h", title="Voltage / V")

        # --- helper (safe heatmap replacement for contourf) ---
        function safe_heatmap(position, data, title_str)
            x, Z, xlabel_str = dashboard_profile(states, position, data)

            x = ustrip.(x)
            Z = Array(ustrip.(Z))

            nt = length(t)
            nx = length(x)

            # fix orientation
            if size(Z) == (nt, nx)
                Z = permutedims(Z)
            elseif size(Z) == (nx, nt)
                # ok
            else
                @warn "Skipping invalid data" size(Z)
                return plot(title="Invalid")
            end

            # clean NaN/Inf
            Z .= ifelse.(isfinite.(Z), Z, 0.0)

            # --- find non-zero spatial region ---
            col_activity = vec(sum(abs.(Z), dims=2))   # sum over time
            
            nonzero_idx = findall(col_activity .> 0)
            
            if isempty(nonzero_idx)
                @warn "All data is zero, skipping node"
                return plot(title="Empty")
            end
            
            xmin_idx = minimum(nonzero_idx)
            xmax_idx = maximum(nonzero_idx)
            
            # --- crop to active region ---
            x = x[xmin_idx:xmax_idx]
            Z = Z[xmin_idx:xmax_idx, :]


            # avoid constant crash
            if abs(maximum(Z) - minimum(Z)) < 1e-12
                Z .+= 1e-6
            end

            return heatmap(x, t, Z';
                title=title_str,
                xlabel=xlabel_str,
                ylabel="Time / h",
                color=:viridis
            )
        end

        # --- 3x2 grid ---
        p1 = safe_heatmap(NeAm_pos, NeAm_conc,
            "NeAm Surface Concentration / mol·m⁻³")

        p2 = safe_heatmap(Elyte_pos, Elyte_conc,
            "Electrolyte Concentration / mol·m⁻³")

        p3 = safe_heatmap(PeAm_pos, PeAm_conc,
            "PeAm Surface Concentration / mol·m⁻³")

        p4 = safe_heatmap(NeAm_pos, NeAm_pot,
            "NeAm Potential / V")

        p5 = safe_heatmap(Elyte_pos, Elyte_pot,
            "Electrolyte Potential / V")

        p6 = safe_heatmap(PeAm_pos, PeAm_pot,
            "PeAm Potential / V")

        return plot(
            p_current,
            p_voltage,
            p1, p2, p3,
            p4, p5, p6;
            layout = @layout([
                a{0.15h};
                b{0.15h};
                [c d e];
                [f g h]
            ]),
            size = (1200, 1000)
        )

    else
        error("Unsupported plot_type: $plot_type")
    end
end

function dashboard_profile(states, position, data)
    if haskey(states, "Cell") && haskey(states["Cell"], "Position")
        cell_position = states["Cell"]["Position"]
        if cell_position isa AbstractVector && size(data, 2) == length(cell_position)
            return (cell_position * 10.0^6, data, "Position  /  μm")
        end
    end

    if position isa AbstractVector && size(data, 2) == length(position)
        return (position * 10.0^6, data, "Position  /  μm")
    end

    x = component_x_coordinates(position)
    if size(data, 2) != length(x)
        error("Could not match position information to data with shape $(size(data)).")
    end
    return collapse_profile_to_x(x, data)
end

function component_x_coordinates(position)
    pp = physical_representation(position)
    primitives = plot_primitives(pp, :meshscatter)
    return primitives.points[:, 1]
end

function collapse_profile_to_x(x_raw, data)
    order = sortperm(x_raw)
    x_sorted = x_raw[order]
    data_sorted = data[:, order]
    n_steps = size(data_sorted, 1)

    x_vals = Float64[]
    columns = Vector{Vector{Float64}}()
    i = 1
    while i <= length(x_sorted)
        j = i
        xi = round(x_sorted[i], sigdigits = 8)
        while j < length(x_sorted) && round(x_sorted[j + 1], sigdigits = 8) == xi
            j += 1
        end
        push!(x_vals, mean(x_sorted[i:j]) * 10.0^6)
        block = data_sorted[:, i:j]
        push!(columns, vec(sum(block, dims = 2)) ./ size(block, 2))
        i = j + 1
    end

    profile = hcat(columns...)'
    return (x_vals, profile', "x-position  /  μm")
end


###############################
# Plot 3D mesh

function extract_edges(grid)
    pts, tri, _ = Jutul.triangulate_mesh(grid; outer=true)
    edge_count = Dict{Tuple{Int,Int}, Int}()
    for i in 1:size(tri,1)
        t = tri[i,:]
        edges = [(t[1],t[2]), (t[2],t[3]), (t[3],t[1])]
        for (a,b) in edges
            if a > b
                a, b = b, a
            end
            edge_count[(a,b)] = get(edge_count, (a,b), 0) + 1
        end
    end
    return pts, edge_count
end

function build_plot(grids; include=nothing, zscale=1.0, camera=(45, 30))
    
    plt = plot(
        xlabel = "x (cm)",
        ylabel = "y (cm)",
        zlabel = "z (μm)",
        grid = false,
        aspect_ratio = :equal,
        size = (900, 700),
        camera = camera   # ✅ added here
    )


    colors = Dict(
        "NegativeElectrodeActiveMaterial" => RGB(0.835, 0.369, 0.0),
        "Separator" => RGB(0.337, 0.706, 0.914),
        "Electrolyte" => RGB(0.0, 0.447, 0.698),
        "PositiveElectrodeActiveMaterial" => RGB(0.0, 0.620, 0.451),
        "NegativeElectrodeCurrentCollector" => RGB(0.800, 0.475, 0.655),
        "PositiveElectrodeCurrentCollector" => RGB(0.941, 0.894, 0.259)
    )

    for (name, grid) in grids
        if include !== nothing && !(name in include)
            continue
        end
        try
            pts, edge_count = extract_edges(grid)

            # ✅ convert units
            x_pts = pts[:,1] .* 100            # m → cm
            y_pts = pts[:,2] .* 100
            z_pts = pts[:,3] .* 1e6 #.* zscale # m → μm

            X = Float64[]
            Y = Float64[]
            Z = Float64[]

            for ((a,b), count) in edge_count
                if count == 1
                    push!(X, x_pts[a]); push!(X, x_pts[b]); push!(X, NaN)
                    push!(Y, y_pts[a]); push!(Y, y_pts[b]); push!(Y, NaN)
                    push!(Z, z_pts[a]); push!(Z, z_pts[b]); push!(Z, NaN)
                end
            end

            plot!(
                plt, X, Y, Z;
                color = get(colors, name, RGB(0.3,0.3,0.3)),
                linewidth = 1.4,
                alpha = 0.95,
                label = name
            )
        catch
            @warn "Skipping $name"
        end
    end

    return plt
end

function plot_grid(grids; include=nothing, zscale_visual=50, camera=(45, 30))
    p = build_plot(grids;
        include=include,
        zscale=zscale_visual,
        camera=camera   # ✅ pass it through
    )

    title!(
        p,
        "Geometry grid"
    )

    return p
end




#########################################
# 2D data plotting


function plot_cell_data_2d(
    grids,
    output;

    variable = "SurfaceConcentration",
    timestep = nothing,
    include = nothing,
    colormap = :viridis,
    unit = ""   # ← NEW
)

    # ------------------------------------------------------------
    # STATE ACCESS
    # ------------------------------------------------------------

    state = output.states

    # ------------------------------------------------------------
    # EXTRACT DATA
    # ------------------------------------------------------------

    data_dict = Dict{String, Vector{Float64}}()

    for domain in keys(state)

        domain_content = state[domain]

        if !(domain_content isa Dict)
            continue
        end

        for submodel in keys(domain_content)

            sub = domain_content[submodel]

            if !(sub isa Dict)
                continue
            end

            if haskey(sub, variable)

                val = sub[variable]

                key = string(domain, submodel)

                # ------------------------------------------------
                # TIME HANDLING
                # ------------------------------------------------

                if val isa AbstractMatrix

                    tidx = timestep === nothing ? size(val,1) : timestep

                    data_dict[key] = vec(val[tidx, :])

                elseif val isa AbstractVector

                    data_dict[key] = vec(val)
                end
            end
        end
    end

    isempty(data_dict) && error("No data found for $variable")

    # ------------------------------------------------------------
    # GLOBAL COLOR RANGE
    # ------------------------------------------------------------

    all_vals = reduce(vcat, values(data_dict))
    clim = extrema(all_vals)

    cmap = cgrad(colormap)

    # ------------------------------------------------------------
    # FIGURE SETUP (2D ONLY)
    # ------------------------------------------------------------

    plt = plot(
        xlabel = "x (cm)",
        ylabel = "y (cm)",
        aspect_ratio = :equal,
        grid = false,
        size = (900, 700)
    )

    # ------------------------------------------------------------
    # LOOP DOMAINS
    # ------------------------------------------------------------

    for (name, grid) in grids

        if include !== nothing && !(name in include)
            continue
        end

        if !haskey(data_dict, name)
            continue
        end

        data = data_dict[name]

        try
            pts, tri, mapper = Jutul.triangulate_mesh(grid; outer=false)

            nc = Jutul.number_of_cells(grid)

            if length(data) != nc
                @warn "Skipping $name: size mismatch"
                continue
            end

            color_vals = mapper.Cells(data)

            # ----------------------------------------------------
            # REMOVE INVALID TRIANGLES
            # ----------------------------------------------------

            keep_tri = Int[]

            for i in axes(tri,1)
                if all(isfinite.(color_vals[tri[i,:]]))
                    push!(keep_tri, i)
                end
            end

            tri = tri[keep_tri, :]

            # ----------------------------------------------------
            # UNIT CONVERSION (2D ONLY)
            # ----------------------------------------------------

            x = pts[:,1] .* 100
            y = pts[:,2] .* 100

            # ----------------------------------------------------
            # TRIANGLE PLOTTING
            # ----------------------------------------------------

            first = true

            for i in axes(tri,1)

                t = tri[i,:]

                xs = [x[t[1]], x[t[2]], x[t[3]], x[t[1]]]
                ys = [y[t[1]], y[t[2]], y[t[3]], y[t[1]]]

                vals = color_vals[t]

                cval = (vals[1] + vals[2] + vals[3]) / 3

                color = cmap[(cval - clim[1]) / (clim[2] - clim[1])]

                plot!(
                    plt,
                    xs, ys;
                    seriestype = :shape,
                    fillcolor = color,
                    linecolor = RGBA(0,0,0,0),
                    fillalpha = 1.0,
                    label = first ? name : nothing
                )

                first = false
            end

        catch err
            @warn "Skipping $name" err
        end
    end

    # ------------------------------------------------------------
    # COLORBAR (WITH UNIT)
    # ------------------------------------------------------------

    colorbar_label = isempty(unit) ? variable : "$variable [$unit]"

    scatter!(
        plt,
        [NaN], [NaN];
        marker_z = [clim[1], clim[2]],
        c = colormap,
        colorbar = true,
        colorbar_title = colorbar_label,
        markersize = 0,
        label = ""
    )

    title!(plt, "$variable (2D slice view)")

    return plt
end