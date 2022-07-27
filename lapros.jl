### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 0911e262-5124-4aa9-b1ba-feacec87b49e
begin
	using Chain
	using DataFrames
	using DataStructures
	import Random
	using SparseArrays: spzeros
	using StatsBase: sample
	using Statistics: mean

	Random.seed!(333)
end;

# ╔═╡ 34e904a0-5909-4cdd-a025-068dbb1a6cd1
n = 10^7 # number of samples

# ╔═╡ 9cb3568a-9b59-438e-9ca0-c10c93247d92
md"""

## Dữ liệu mẫu

Lấy ví dụ cụ thể 
đối với $n mẫu dữ liệu
ta có các phân lớp,
nhãn quan sát và xác suất dự đoán 
như sau. 
Đặt mục tiêu viết thuật toán tìm nhãn lỗi chạy trong vòng 1 giây.
"""

# ╔═╡ dc3a9ad9-6a90-437b-9893-d59788b8bafb
m = 3 # number of classes

# ╔═╡ d1dd37af-7def-4a27-add4-223bf76757e2
M = collect(1:m)

# ╔═╡ 8c9ff1af-26ef-42b4-8e1b-39d978b08674
ỹ = sample(M, n)

# ╔═╡ 99e5c1b4-89a0-4e70-972e-37dd22d57d37
p̂ = @chain rand(Float64, (n, m)) begin
	_ ./ sum(_, dims=2)
	# _[:, 2:end]
end

# ╔═╡ bbb3b95f-e45f-46b6-b6e6-b41b2b283ab2
@assert all(sum(p̂, dims=2) .≈ 1)

# ╔═╡ 1a048638-4eda-4d6c-9009-6717a330aeb6
md"""

## TLDR

Xếp hạng độ khả nghi của các nhãn là như sau.
"""

# ╔═╡ 1dbe808e-85e0-4052-b6ac-90fa86300ba7
md"""
Nếu không chỉ đỉnh `classes` thì dùng mặc định là các giá trị unique của `observed`. 
"""

# ╔═╡ a6a56ac9-6849-49f9-bbce-4222cf663c80
function lapros(
	p::Array{Float64,2}, 
	y::Array{Int64,1}, 
)
	M::Array{Int64,1} = unique(y)
	M = sort(M)
	# @show M
	lapros(p, y, M)
end

# ╔═╡ faa8ebef-ff1a-486d-926f-3ab1a782eaf1
md"""

## Đếm nhãn

Tập hợp mẫu quan sát được cho từng lớp là
"""

# ╔═╡ 95d29260-af27-48ca-a3a7-120575b318a4
Xỹ = [(1:n)[ỹ .== i] for i ∈ M]

# ╔═╡ 0327985e-859f-4c0e-a232-856aa31b0c44
md"""
Ta đếm được số lượng mẫu quan sát được cho từng lớp là
"""

# ╔═╡ 846072fc-2269-4fbe-89d4-270b71cf5339
C = [sum(ỹ .== i) for i ∈ M]

# ╔═╡ 68fef702-a604-4f03-87da-09984af59898
md"""

## Chỉ tiêu tự tin

Độ tự tin trung bình của từng lớp là
"""

# ╔═╡ 6c190eec-b8bc-4915-9d20-cac0280b4131
md"""
Mức độ vượt chỉ tiêu của từng xác suất dự đoán là 
"""

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

# ╔═╡ 591f5a84-b437-4b59-9961-bc110e34eeae
"Find the most likely labels for each sample.

## Params:

- mask_negative: For some specific sample, if the normalized probabilities are all negative then we use 0 to mark that there is no likely class label for the sample."
function find_likely_label(t2p::Array{Float64,2}, mask_negative::Bool=false)
	am = argmax(t2p, dims=2)[:]
	likely_labels = last.(Tuple.(am))
	if mask_negative
		ifelse.(any(t2p .≥ 0, dims=2)[:], likely_labels, 0)
	else
		likely_labels
	end
end

# ╔═╡ 451f8698-2c6c-4cf1-8e1a-ede8cc1c535b
md"""
Xếp các mẫu vào ma trận có hàng thể hiện nhãn đã quan sát $\tilde{y}$, còn cột thể hiện nhãn đáng tin nhất $\hat{l}.$
"""

# ╔═╡ 45b1feb2-45d3-46df-973a-8b95a70ea164
# Xỹẏ = partition_X(l̂, ỹ, M)

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

# ╔═╡ b33c68aa-0a24-4622-8e4d-d9abbce885a5
function rank_suspicious(
	ps::Array{Float64,2}, 
	ls::Array{Int64,1},
	ys::Array{Int64,1},
)
	# @show ps
	# @show ls
	# @show ys
	n = length(ys)
	e = spzeros(Float64, n)	
	# @show e
	ids = (ls.≠ys) .&& (ls.≠0)
	for k in (1:n)[ids]
		e[k] = ps[k, ls[k]] - ps[k, ys[k]] 
	end
	# @show e
	e
end

# ╔═╡ 8fadfa66-a730-4f99-aa70-e25fec26cb09
function avg_confidence(
	ps::Array{Float64,2}, 
	ys::Array{Int64,1}, 
	M::Array{Int64,1},
)
	t = [mean(ps[ys .== i, i]) for i ∈ M]'
end

# ╔═╡ df59e679-7e5d-4c3e-b49d-6cd899ca55d9
"Rank the suspiciouness of observed labels.

Parameters: 
- `p`: predicted probabilities given by some model
- `y`: observed labels of the samples
- `M`: unique classes of possible labels

If not specified, `M` will be computed by taking unique values of `y` and sort them.
"
function lapros(
	p::Array{Float64,2}, 
	y::Array{Int64,1}, 
	M::Array{Int64,1},
)
	t = avg_confidence(p, y, M)
	ll = find_likely_label(p .- t)
	rank = rank_suspicious(p, ll, y)
end

# ╔═╡ 2c9a31f4-6abd-40ef-a3b3-f6eb39eda059
errs = lapros(p̂, ỹ, M)

# ╔═╡ 026a4124-a6df-4ae3-a23b-fd676fd3b2cb
errs_ = lapros(p̂, ỹ)

# ╔═╡ b8555fa5-d21c-44da-832c-1366f304cde4
@assert all(errs .≡ errs_)

# ╔═╡ 1dba3ac3-eb7c-442c-a474-cd7b415a4594
t = avg_confidence(p̂, ỹ, M)

# ╔═╡ 154ea90b-578c-4348-be28-69a3877e48d0
sum(t)

# ╔═╡ 71c14b17-6fda-4b0a-b30f-c3c181d35b12
t2p = p̂ .- t

# ╔═╡ 5921b759-9474-414f-a25a-416aec299afb
l̂ = find_likely_label(t2p)

# ╔═╡ a7dfaa5e-7376-49c7-b025-f16e4c5b8f56
rank_suspicious(t2p, l̂, ỹ)

# ╔═╡ 2dbdc102-5f61-4888-9bae-96d8bbf333a5
# Ma trận True/False thể hiện việc xác suất dự đoán có đạt chỉ tiêu hay không:
t2p_positive = t2p .≥ 0;

# ╔═╡ 82974108-2917-41f8-a8b2-0fe323331b48
@assert t2p_positive == (p̂ .≥ t)

# ╔═╡ 29cd589c-b3e9-4a65-9641-352129a27c1e
# using PlutoUI; TableOfContents()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
Chain = "~0.5.0"
DataFrames = "~1.3.4"
DataStructures = "~0.18.13"
StatsBase = "~0.33.19"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Chain]]
git-tree-sha1 = "8c4920235f6c561e401dfe569beb8b924adad003"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.5.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

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

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

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

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

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

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

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

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2c11d7290036fe7aac9038ff312d3b3a2a5bf89e"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.4.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "472d044a1c8df2b062b23f222573ad6837a615ba"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.19"

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

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

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
# ╠═34e904a0-5909-4cdd-a025-068dbb1a6cd1
# ╟─dc3a9ad9-6a90-437b-9893-d59788b8bafb
# ╟─d1dd37af-7def-4a27-add4-223bf76757e2
# ╟─8c9ff1af-26ef-42b4-8e1b-39d978b08674
# ╟─99e5c1b4-89a0-4e70-972e-37dd22d57d37
# ╠═bbb3b95f-e45f-46b6-b6e6-b41b2b283ab2
# ╟─1a048638-4eda-4d6c-9009-6717a330aeb6
# ╠═2c9a31f4-6abd-40ef-a3b3-f6eb39eda059
# ╠═df59e679-7e5d-4c3e-b49d-6cd899ca55d9
# ╟─1dbe808e-85e0-4052-b6ac-90fa86300ba7
# ╠═b8555fa5-d21c-44da-832c-1366f304cde4
# ╠═026a4124-a6df-4ae3-a23b-fd676fd3b2cb
# ╠═a6a56ac9-6849-49f9-bbce-4222cf663c80
# ╟─faa8ebef-ff1a-486d-926f-3ab1a782eaf1
# ╠═95d29260-af27-48ca-a3a7-120575b318a4
# ╟─0327985e-859f-4c0e-a232-856aa31b0c44
# ╟─846072fc-2269-4fbe-89d4-270b71cf5339
# ╟─68fef702-a604-4f03-87da-09984af59898
# ╠═1dba3ac3-eb7c-442c-a474-cd7b415a4594
# ╠═154ea90b-578c-4348-be28-69a3877e48d0
# ╟─6c190eec-b8bc-4915-9d20-cac0280b4131
# ╠═71c14b17-6fda-4b0a-b30f-c3c181d35b12
# ╟─73ef22d7-0961-46fa-93a8-c149c62f0ba8
# ╟─21b37653-4dc8-48c6-bc31-cd0449b24bc2
# ╠═f8cbcd15-d2d0-4945-a80b-0e0e615d22e5
# ╟─5be2ee06-178d-42ec-97b2-b51dbe189633
# ╠═5921b759-9474-414f-a25a-416aec299afb
# ╠═591f5a84-b437-4b59-9961-bc110e34eeae
# ╟─451f8698-2c6c-4cf1-8e1a-ede8cc1c535b
# ╠═45b1feb2-45d3-46df-973a-8b95a70ea164
# ╠═2a49d3be-d6ae-43fd-8fe8-a14800c023c9
# ╟─9c063c91-0497-40c2-8eb1-bdf8060aaf70
# ╠═a7dfaa5e-7376-49c7-b025-f16e4c5b8f56
# ╠═b33c68aa-0a24-4622-8e4d-d9abbce885a5
# ╠═8fadfa66-a730-4f99-aa70-e25fec26cb09
# ╠═2dbdc102-5f61-4888-9bae-96d8bbf333a5
# ╠═82974108-2917-41f8-a8b2-0fe323331b48
# ╠═0911e262-5124-4aa9-b1ba-feacec87b49e
# ╟─29cd589c-b3e9-4a65-9641-352129a27c1e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
