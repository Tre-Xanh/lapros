### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 0911e262-5124-4aa9-b1ba-feacec87b49e
begin
	import Arrow
	using Chain: @chain
	import CSV
	using DataFrames
	using DataStructures
	using Formatting: format
	using SparseArrays: spzeros, findnz
	using Statistics: mean
end;

# ╔═╡ 29cd589c-b3e9-4a65-9641-352129a27c1e
using PlutoUI; TableOfContents()

# ╔═╡ e019b221-ff69-448b-baf3-99e142368a5f
(data_fp, output_fp) = (
	("./data/input/lapros-testdata-n10-m3.csv", "./data/output/lapros-rankdata-n10-m3.csv")
	# ("./data/input/lapros-testdata-n10000000-m3.arrow", "./data/output/lapros-rankdata-n10000000-m3.arrow")
)

# ╔═╡ 1800ac8f-683d-416e-aef1-802af62ccc81
df = if endswith(data_fp, ".csv")
		DataFrame(CSV.File(data_fp))
	else
		DataFrame(Arrow.Table(data_fp))
	end

# ╔═╡ 34e904a0-5909-4cdd-a025-068dbb1a6cd1
n = size(df, 1) # number of samples

# ╔═╡ dc3a9ad9-6a90-437b-9893-d59788b8bafb
m = size(df, 2) - 1 # number of classes

# ╔═╡ 9cb3568a-9b59-438e-9ca0-c10c93247d92
md"""

## Dữ liệu mẫu

Lấy ví dụ cụ thể
đối với $(format(n, commas = true)) mẫu dữ liệu
trong $(m) phân lớp,
đã được gán nhãn quan sát và có xác suất dự đoán từ một mô hình nào đó.
Đặt mục tiêu viết thuật toán tìm nhãn lỗi chạy trong vòng 1 giây.
"""

# ╔═╡ 8c9ff1af-26ef-42b4-8e1b-39d978b08674
ỹ = Array(df[!, :label])

# ╔═╡ 99e5c1b4-89a0-4e70-972e-37dd22d57d37
p̂ = Matrix(df[!, Not(:label)])

# ╔═╡ bbb3b95f-e45f-46b6-b6e6-b41b2b283ab2
@assert all(sum(p̂, dims=2) .≈ 1)

# ╔═╡ 1a048638-4eda-4d6c-9009-6717a330aeb6
md"""

## TLDR

Xếp hạng độ khả nghi của các nhãn là như sau.
"""

# ╔═╡ faa8ebef-ff1a-486d-926f-3ab1a782eaf1
md"""

## Đếm nhãn

Tập hợp mẫu quan sát được cho từng lớp là
"""

# ╔═╡ 95d29260-af27-48ca-a3a7-120575b318a4
Xỹ = [(1:n)[ỹ .== i] for i ∈ 1:m]

# ╔═╡ 0327985e-859f-4c0e-a232-856aa31b0c44
md"""
Ta đếm được số lượng mẫu quan sát được cho từng lớp là
"""

# ╔═╡ 846072fc-2269-4fbe-89d4-270b71cf5339
C = [sum(ỹ .== i) for i ∈ 1:m]

# ╔═╡ 68fef702-a604-4f03-87da-09984af59898
md"""

## Chỉ tiêu tự tin

Độ tự tin trung bình của từng lớp là
"""

# ╔═╡ 71395450-5a24-4bfb-bcf9-b17ae0f626d3
p̂

# ╔═╡ 8fadfa66-a730-4f99-aa70-e25fec26cb09
"Compute the average model confidence for samples in each class.
If there is no sample in some specific class,
we take the avarage over all samples."
function avg_confidence(
	ps::Array{Float64,2},
	ys::Array{Int64,1},
)
	# @show ys
	m = size(ps, 2)
	t = zeros(Float64, m)
	for i ∈ 1:m
		x_yi = ys .== i
		# @show i, x_yi
		if any(x_yi)
			t[i] = mean(ps[x_yi, i])
		else
			t[i] = mean(ps[:, i])
		end
	end
	t
	# @show t
end

# ╔═╡ 1dba3ac3-eb7c-442c-a474-cd7b415a4594
@time t = avg_confidence(p̂, ỹ)

# ╔═╡ 154ea90b-578c-4348-be28-69a3877e48d0
sum(t)

# ╔═╡ 6c190eec-b8bc-4915-9d20-cac0280b4131
md"""
Mức độ vượt chỉ tiêu của từng xác suất dự đoán là
"""

# ╔═╡ 71c14b17-6fda-4b0a-b30f-c3c181d35b12
t2p = p̂ .- t'

# ╔═╡ 73ef22d7-0961-46fa-93a8-c149c62f0ba8
md""" ## Chọn nhãn khả tín
"""

# ╔═╡ 21b37653-4dc8-48c6-bc31-cd0449b24bc2
md"""
Ta có các tập nhãn khả tín như sau.
"""

# ╔═╡ f8cbcd15-d2d0-4945-a80b-0e0e615d22e5
# Lᵩ = [M[t2p_positive[i, :]] for i in 1:n]

# ╔═╡ 5be2ee06-178d-42ec-97b2-b51dbe189633
md"""
Danh sách nhãn đáng tin nhất đối với từng mẫu là như sau, trong đó $0$ đánh dấu trường hợp không có nhãn phù hợp.
"""

# ╔═╡ 69e425b0-7b6b-4c2f-afba-70ea787bde54
t2p

# ╔═╡ 591f5a84-b437-4b59-9961-bc110e34eeae
"Find the most likely labels for each sample.

## Params:

- mask_negative: For some specific sample, if the normalized probabilities are all negative then we use 0 to mark that there is no likely class label for the sample."
function find_likely_label(t2p::Array{Float64,2}, mask_negative::Bool=false)
	# @show t2p
	am = argmax(t2p, dims=2)
	# @show am
	# @time am = am[:]
	# @show am
	likely_labels = last.(Tuple.(am))
	ll = if mask_negative
		ifelse.(any(t2p .≥ 0, dims=2)[:], likely_labels, 0)
	else
		likely_labels
	end
	vec(ll)
	# @show ll
end

# ╔═╡ 5921b759-9474-414f-a25a-416aec299afb
@time l̂ = find_likely_label(t2p)

# ╔═╡ 451f8698-2c6c-4cf1-8e1a-ede8cc1c535b
md"""
Xếp các mẫu vào ma trận có hàng thể hiện nhãn đã quan sát $\tilde{y}$, còn cột thể hiện nhãn đáng tin nhất $\hat{l}.$
"""

# ╔═╡ 45b1feb2-45d3-46df-973a-8b95a70ea164
# Xỹẏ = partition_X(l̂, ỹ, 1:m)

# ╔═╡ 2a49d3be-d6ae-43fd-8fe8-a14800c023c9
function partition_X(
	ls::Array{Int64,1},
	ys::Array{Int64,1},
	M::Array{Int64,1})
	# @show ys
	# @show ls
	X_partition = [[] for i ∈ M, j ∈ M]
	for (x, (i,j)) ∈ enumerate(zip(ys,ls))
		if j ∈ M
			push!(X_partition[i,j], x)
		end
	end
	X_partition
end

# ╔═╡ 9c063c91-0497-40c2-8eb1-bdf8060aaf70
md"""
## Độ khả nghi

Độ khả nghi của các mẫu dữ liệu là như sau.
"""

# ╔═╡ 14d5fa77-7393-41ef-95e2-6848976c3346
t2p

# ╔═╡ f44f3c50-2c8f-4ea6-b712-07fa9f951ffb
function rank_suspicious(
	ps::Array{Float64,2},
	ls::Array{Int64,1},
	ys::Array{Int64,1}
)
	# @show ls
	# @show ys
	n,m = size(ps)
	e = spzeros(Float64, n)
	ids = (ls.≠ys) .&& (ls.≠0)
	for k in (1:n)[ids]
		e[k] = ps[k, ls[k]] - ps[k, ys[k]]
		# @show k, ls[k], ys[k], e[k]
	end
	# @show e
	e
end

# ╔═╡ df59e679-7e5d-4c3e-b49d-6cd899ca55d9
"Rank the suspiciouness of observed labels.

Parameters:
- `p`: predicted probabilities given by some model
- `y`: observed labels of the samples
"
function lapros(
	p::Array{Float64,2},
	y::Array{Int64,1}
)
	t = avg_confidence(p, y)
	t2p = p .- t'
	@time ll = find_likely_label(t2p)
	@time rank = rank_suspicious(t2p, ll, y)
end

# ╔═╡ 2c9a31f4-6abd-40ef-a3b3-f6eb39eda059
@time errs = lapros(p̂, ỹ)

# ╔═╡ b3409c7f-a2bd-4ede-b857-17a6aba925f8
@assert all(errs .≥ 0)

# ╔═╡ a7dfaa5e-7376-49c7-b025-f16e4c5b8f56
@time rank = rank_suspicious(t2p, l̂, ỹ)

# ╔═╡ 9bb332cb-fc80-42b3-a4d9-e4354bb4720f
@assert all(errs .≈ rank)

# ╔═╡ 7edc3678-03fe-4eb8-ab39-f7033a06e93d
# begin
# 	@time rank_v1 = rank_suspicious_v1(t2p, l̂, ỹ)
# 	@assert all(rank .≡ rank_v1)
# end

# ╔═╡ b33c68aa-0a24-4622-8e4d-d9abbce885a5
function rank_suspicious_v1(
	ps::Array{Float64,2},
	ls::Array{Int64,1},
	ys::Array{Int64,1}
)
	# @show ls
	# @show ys
	n,m = size(ps)
	e = spzeros(Float64, n)
	ids = (ls.≠ys) .&& (ls.≠0)
	# @show ids
	# @show (1:n)[ids]
	for j in 1:m
		# @show j
		idls = (1:n)[ids .&& (ls.≡j)]
		# @show idls
		e[idls] += ps[idls, j]
		idys = (1:n)[ids .&& (ys.≡j)]
		# @show idys
		e[idys] -= ps[idys, j]
		# @show e
	end
	e
end

# ╔═╡ f0d78c9e-1798-4e68-bb2f-a83fdb4fb531
md"""
# Save ranks
"""

# ╔═╡ 759c38c1-9e20-4fd0-8038-7c3287af27e6
(err_id, err_val) = findnz(errs)

# ╔═╡ ee56ca9c-2458-48a1-bebf-aea0946efc11
df_err = DataFrame(id=err_id, err=err_val)

# ╔═╡ 05f3037a-9d08-49ea-b06a-28a78ab7ea9c
if endswith(output_fp, ".csv")
	CSV.write(output_fp, df_err)
else
	Arrow.write(output_fp, df_err)
end

# ╔═╡ 31d9745d-4d9c-4cf9-8bfe-304c74320f62
size(df_err, 1)/n

# ╔═╡ afd7734d-a109-46f1-a982-b348520bcfe1
md"""
## Tools
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
Formatting = "59287772-0a20-5a39-b81b-1366585eb4c0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Arrow = "~2.3.0"
CSV = "~0.10.4"
Chain = "~0.5.0"
DataFrames = "~1.3.4"
DataStructures = "~0.18.13"
Formatting = "~0.4.2"
PlutoUI = "~0.7.39"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arrow]]
deps = ["ArrowTypes", "BitIntegers", "CodecLz4", "CodecZstd", "DataAPI", "Dates", "Mmap", "PooledArrays", "SentinelArrays", "Tables", "TimeZones", "UUIDs"]
git-tree-sha1 = "4e7aa2021204bd9456ad3e87372237e84ee2c3c1"
uuid = "69666777-d1a9-59fb-9406-91d4454c9d45"
version = "2.3.0"

[[deps.ArrowTypes]]
deps = ["UUIDs"]
git-tree-sha1 = "a0633b6d6efabf3f76dacd6eb1b3ec6c42ab0552"
uuid = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
version = "1.2.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitIntegers]]
deps = ["Random"]
git-tree-sha1 = "5a814467bda636f3dde5c4ef83c30dd0a19928e0"
uuid = "c3b6d118-76ef-56ca-8cc7-ebb389d030a1"
version = "0.2.6"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[deps.Chain]]
git-tree-sha1 = "8c4920235f6c561e401dfe569beb8b924adad003"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.5.0"

[[deps.CodecLz4]]
deps = ["Lz4_jll", "TranscodingStreams"]
git-tree-sha1 = "59fe0cb37784288d6b9f1baebddbf75457395d40"
uuid = "5ba52731-8f18-5e0d-9241-30f10d1ec561"
version = "0.4.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.CodecZstd]]
deps = ["CEnum", "TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "849470b337d0fa8449c21061de922386f32949d9"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.7.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "d19f9edd8c34760dca2de2b503f969d8700ed288"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "29714d0a7a8083bba8427a4fbfb00a540c681ce7"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimeZones]]
deps = ["Dates", "Downloads", "InlineStrings", "LazyArtifacts", "Mocking", "Printf", "RecipesBase", "Scratch", "Unicode"]
git-tree-sha1 = "d634a3641062c040fc8a7e2a3ea17661cc159688"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.9.0"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─9cb3568a-9b59-438e-9ca0-c10c93247d92
# ╠═e019b221-ff69-448b-baf3-99e142368a5f
# ╠═1800ac8f-683d-416e-aef1-802af62ccc81
# ╟─34e904a0-5909-4cdd-a025-068dbb1a6cd1
# ╟─dc3a9ad9-6a90-437b-9893-d59788b8bafb
# ╠═8c9ff1af-26ef-42b4-8e1b-39d978b08674
# ╟─99e5c1b4-89a0-4e70-972e-37dd22d57d37
# ╠═bbb3b95f-e45f-46b6-b6e6-b41b2b283ab2
# ╟─1a048638-4eda-4d6c-9009-6717a330aeb6
# ╠═2c9a31f4-6abd-40ef-a3b3-f6eb39eda059
# ╠═b3409c7f-a2bd-4ede-b857-17a6aba925f8
# ╠═9bb332cb-fc80-42b3-a4d9-e4354bb4720f
# ╠═df59e679-7e5d-4c3e-b49d-6cd899ca55d9
# ╟─faa8ebef-ff1a-486d-926f-3ab1a782eaf1
# ╠═95d29260-af27-48ca-a3a7-120575b318a4
# ╟─0327985e-859f-4c0e-a232-856aa31b0c44
# ╟─846072fc-2269-4fbe-89d4-270b71cf5339
# ╟─68fef702-a604-4f03-87da-09984af59898
# ╠═1dba3ac3-eb7c-442c-a474-cd7b415a4594
# ╠═71395450-5a24-4bfb-bcf9-b17ae0f626d3
# ╠═8fadfa66-a730-4f99-aa70-e25fec26cb09
# ╠═154ea90b-578c-4348-be28-69a3877e48d0
# ╟─6c190eec-b8bc-4915-9d20-cac0280b4131
# ╠═71c14b17-6fda-4b0a-b30f-c3c181d35b12
# ╟─73ef22d7-0961-46fa-93a8-c149c62f0ba8
# ╟─21b37653-4dc8-48c6-bc31-cd0449b24bc2
# ╠═f8cbcd15-d2d0-4945-a80b-0e0e615d22e5
# ╟─5be2ee06-178d-42ec-97b2-b51dbe189633
# ╠═69e425b0-7b6b-4c2f-afba-70ea787bde54
# ╠═5921b759-9474-414f-a25a-416aec299afb
# ╠═591f5a84-b437-4b59-9961-bc110e34eeae
# ╟─451f8698-2c6c-4cf1-8e1a-ede8cc1c535b
# ╠═45b1feb2-45d3-46df-973a-8b95a70ea164
# ╠═2a49d3be-d6ae-43fd-8fe8-a14800c023c9
# ╟─9c063c91-0497-40c2-8eb1-bdf8060aaf70
# ╠═a7dfaa5e-7376-49c7-b025-f16e4c5b8f56
# ╠═14d5fa77-7393-41ef-95e2-6848976c3346
# ╠═f44f3c50-2c8f-4ea6-b712-07fa9f951ffb
# ╠═7edc3678-03fe-4eb8-ab39-f7033a06e93d
# ╠═b33c68aa-0a24-4622-8e4d-d9abbce885a5
# ╟─f0d78c9e-1798-4e68-bb2f-a83fdb4fb531
# ╠═759c38c1-9e20-4fd0-8038-7c3287af27e6
# ╠═ee56ca9c-2458-48a1-bebf-aea0946efc11
# ╠═05f3037a-9d08-49ea-b06a-28a78ab7ea9c
# ╠═31d9745d-4d9c-4cf9-8bfe-304c74320f62
# ╟─afd7734d-a109-46f1-a982-b348520bcfe1
# ╠═0911e262-5124-4aa9-b1ba-feacec87b49e
# ╠═29cd589c-b3e9-4a65-9641-352129a27c1e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
