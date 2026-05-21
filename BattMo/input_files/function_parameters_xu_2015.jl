using Jutul: get_1d_interpolator

function _open_circuit_potential_graphite_Xu_2015(c, T, refT, cmax)

	"""Compute OCP for LFP as function of temperature and concentration"""
	refT = 298.15
	theta = c ./ cmax

	data1 = [
		0.0 1.28683
		0.01 0.65272
		0.02 0.52621
		0.03 0.44128
		0.04 0.37552
		0.05 0.32567
		0.1 0.21665
		0.15 0.18623
		0.2 0.16445
		0.25 0.14548
		0.3 0.13293
		0.35 0.12635
		0.4 0.123
		0.45 0.12036
		0.5 0.11606
		0.55 0.10811
		0.6 0.09833
		0.65 0.09146
		0.7 0.08829
		0.75 0.08696
		0.8 0.08592
		0.85 0.08369
		0.9 0.07698
		0.95 0.05692
		0.96 0.0498
		0.97 0.04118
		0.98 0.03086
		0.99 0.01865
		1.0 0.00443
	]

	x1 = data1[:, 1]
	y1 = data1[:, 2]

	itp_refOCP = get_1d_interpolator(x1, y1, cap_endpoints = false)

	refOCP = itp_refOCP(theta)

	return refOCP

end


function _open_circuit_potential_lfp_Xu_2015(c, T, refT, cmax)

	"""Compute OCP for LFP as function of temperature and concentration"""
	refT = 298.15
	theta = c ./ cmax

	data1 = [
		0.0 4.1433
		0.01 3.9121
		0.02 3.7272
		0.03 3.606
		0.04 3.5326
		0.05 3.4898
		0.1 3.436
		0.15 3.4326
		0.2 3.4323
		0.25 3.4323
		0.3 3.4323
		0.35 3.4323
		0.4 3.4323
		0.45 3.4323
		0.5 3.4323
		0.55 3.4323
		0.6 3.4323
		0.65 3.4323
		0.7 3.4323
		0.75 3.4323
		0.8 3.4322
		0.85 3.4311
		0.9 3.4142
		0.95 3.2515
		0.96 3.1645
		0.97 3.0477
		0.98 2.8999
		0.99 2.7312
		1.0 2.5895
	]
	x1 = data1[:, 1]
	y1 = data1[:, 2]

	itp_refOCP = get_1d_interpolator(x1, y1, cap_endpoints = false)

	refOCP = itp_refOCP(theta)


	return refOCP

end


function _electrolyte_conductivity_Xu_2015(c::Real, T::Real)
	""" Compute the electrolyte conductivity as a function of concentration
	"""
	conductivityFactor = 1.0e-4

	conductivity = c * 1.0e-4 * 1.2544 * (-8.2488 + 0.053248 * T - 2.987e-5 * (T^2) + 0.26235e-3 * c - 9.3063e-6 * c * T + 8.069e-9 * c * T^2 + 2.2002e-7 * c^2 - 1.765e-10 * T * c^2)
	return conductivity
end

function _electrolyte_diffusivity_Xu_2015(c::Real, T::Real)
	""" Compute the diffusion coefficient as a function of concentration
	"""
	# Calculate diffusion coefficients constant for the diffusion coefficient calculation
	cnst = [
		-4.43  -54.0;
		-0.22   0.0
	]

	Tgi = [229 5.0]

	# Diffusion coefficient, [m^2 s^-1]
	#Removed 10⁻⁴ otherwise the same
	D = 10^((cnst[1, 1] + cnst[1, 2] / (T - Tgi[1] - Tgi[2] * c * 1.0e-3) + cnst[2, 1] * c * 1.0e-3))
	return D
end
