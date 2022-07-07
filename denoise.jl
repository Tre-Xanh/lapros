### A Pluto.jl notebook ###
# v0.19.9

#> [frontmatter]
#> title = "Denoise Labels"

using Markdown
using InteractiveUtils

# ╔═╡ fb4735cc-70fb-443b-97de-def822e0c826
begin
	md"""
	$\newcommand{\vect}[1]{{\boldsymbol{{#1}}}}$
	$\newcommand{\x}{\vect{x}}$
	$\newcommand{\X}{\vect{X}}$
	$\newcommand{\ystar}{y^{*}}$
	$\newcommand{\ytilde}{\tilde{y}}$
	$\newcommand{\argmax}{\mathop{\rm arg~max}\limits}$
	"""
	using PlutoUI; PlutoUI.TableOfContents()
end

# ╔═╡ d83aec0b-4418-425e-a91e-b6d17b1fff69
md"# Tìm lỗi quan sát nhãn dữ liệu

Có một bộ dữ liệu bảng số, được gắn nhãn để phân loại, ví dụ nhãn dương tính và âm tính. Giả sử các nhãn dương tính có độ tin cậy cao, còn các nhãn âm tính có độ tin cậy thấp hơn, có thể xem như chứa cả dữ liệu dương tính chưa bộc phát. Lấy ví dụ với dữ liệu đánh giá tín dụng thì những ca vỡ nợ sẽ có nhãn dương tính. Với dữ liệu khám nghiệm ung thư thì những ca đã phát bệnh là dương tính.

_Giả sử phân bố nhãn trong bộ dữ liệu trên có tương quan đủ mạnh với phân bố nhãn tiềm ẩn thật sự_. Trong phân bố nhãn có tiềm ẩn tính đa dạng, bất định mang tính bản chất. Ví dụ có 10 người có các thuộc tính xấu gần giống nhau, nhưng sẽ trong đó sẽ chỉ có 8 người ngẫu nhiên nào đó vỡ nợ, hoặc bị ung thư.

Bộ dữ liệu trên có một số lượng nhất định các nhãn bị gắn sai. Có thể nào xác định được ranh giới để phân biệt được lỗi gắn nhãn với tính bất định bản chất của dữ liệu hay không?

_Giả sử rằng suất nhãn bị gán sai không phụ thuộc vào từng ca dữ liệu cụ thể mà chỉ phụ thuộc vào đặc tính của các lớp nhãn dữ liệu_. Ví dụ trường hợp phân loại 3 loài vật là chó, mèo và chuột, thì xác suất nhầm chó với mèo cao hơn là nhầm mèo với chuột hoặc chó với chuột.

Có một mô hình dự đoán xác suất dương tính đối với bộ dữ liệu nêu trên. _Giả sử xác suất do mô hình đưa ra có tương quan đủ mạnh đối với phân bố thật sự của nhãn_.

Với những giả sử nêu trên, ta có thể ước lượng được xác suất nhãn gắn trên một ca dữ liệu là thật sự đúng hay không?"

# ╔═╡ 7013e482-ec10-42cc-813f-cd7c29759757
md"""## Định nghĩa và ký hiệu
"""

# ╔═╡ dbd6bcfa-fd0b-11ec-3928-79a4a6776291
md"""### Quy trình nhiễu theo lớp

Giả sử có một bộ dữ liệu số được gắn nhãn phân loại thành $m$ lớp khác nhau $[m] \coloneqq {1,2,\ldots,m}$. 
Giả sử đối với mỗi mẫu dữ liệu ta có một nhãn "tiềm ẩn" thật là $\ystar$.
Trước khi quan sát được nhãn $\ytilde$, giả sử có một quy trình gây nhiễu
biến $\ystar=j$ thành $\ytilde=i$ với xác suất 
$p(\ytilde=i, \ystar=j)$ chỉ phụ thuộc vào $i,j \in [m]$ và độc lập với các mẫu dữ liệu cụ thể, 

$p(\ytilde| \ystar; \vect{x}) = p(\ytilde| \ystar) \forall\vect{x}.$

Ví dụ khi phân loại 3 loài vật là chó, mèo và chuột, thì xác suất nhầm chó với mèo cao hơn là nhầm mèo với chuột hoặc chó với chuột, và xác suất đó không phụ thuộc vào từng con cho, con mèo hoặc con chuột cụ thể. Giả sử này là hợp lý và thường được sử dụng trong các nghiên cứu về xử lý nhiễu (Goldberger and Ben-Reuven, 2017; Sukhbaatar et al., 2015).
"""


# ╔═╡ bc77351a-9d49-4fa6-854d-ed7adba9c300
md"""### Ma trận nhiễu theo lớp
$\boldsymbol{Q}_{\ytilde, \ystar} \coloneqq \left[ {\begin{array}{ccc}
    p(\ytilde=1, \ystar=1) & \ldots & p(\ytilde=1, \ystar=m) \\
    \vdots & p(\ytilde=i, \ystar=j) & \vdots \\
    p(\ytilde=m, \ystar=1) & \ldots & p(\ytilde=m, \ystar=m) \\
  \end{array} } \right]$ 
là ma trận kích thước $m\times m$ thể hiện phân phối xác suất đồng thời cho $\ytilde$ và $\ystar.$

**Độ thưa** là tỷ lệ số $0$ chiếm lĩnh các vị trí ngoại trừ đường chéo của ma trận $\vect{Q}_{\ytilde,\ystar}$: độ thưa bằng $0$ nói rằng mọi tỷ lệ nhiễu $p_{\ytilde,\ystar}$ đều khác $0$, còn độ thưa $1$ thể hiện tình trạng lý tưởng, hoàn toàn không có nhiễu trong nhãn.

Gọi $\X_{\ytilde=i}$ là tập hợp các mẫu $\x$ đã được gán nhãn $\ytilde=i$.
**Độ tự tin** $\hat{p}(\ytilde=i; \vect{x}\in\vect{X}_{\ytilde=i},\vect{\theta})$
là xác suất mô hình $\vect{\theta}$ đưa ra đối với mẫu $\vect{x}$, dự đoán nó có label đúng như label $\ytilde$ đã được gán. *Độ tự tin thấp là một dấu hiệu của khả năng nhãn có lỗi.*

"""

# ╔═╡ b51e3d11-aedc-431d-b591-d808f0a884dc
md"""
## Phương pháp học tự tin

Confident Learning Method

Đầu vào:
- các nhãn $\ytilde_k$ đã quan sát được và gán cho các mẫu $\x_k\in\X$
- dự đoán xác suất $\hat{p}(\ytilde=i; \vect{x}_k\in\vect{X},\vect{\theta})$ do một mô hình phân loại $\vect{\theta}$ đưa ra

Các bước:
0. Tính $t_i$, độ tự tin trung bình  trong từng lớp $i\in[m]$
1. Ước lượng phân bố xác suất đồng thời $\boldsymbol{\hat{Q}}_{\ytilde, \ystar}$ cho nhãn quan sát và nhãn thật
2. Tìm và loại bỏ các mẫu có khả năng nhãn bị lỗi cao
3. Đặt trọng số cho các mẫu trong từng lớp $i\in[m]$ để học lại mô hình $\vect{\theta}$
"""

# ╔═╡ 18943cbd-4045-48cc-9857-e42e86aa8556
md"""
### Chỉ tiêu tự tin

Với mỗi lớp $i\in[m]$ ta có thể chọn một chỉ tiêu tự tin $t_j\in(0,1)$.
Một cách chọn chỉ tiêu tự tin là dùng độ tự tin trung bình $(\ref{eq2})$.
Đối với từng mẫu $\x$ và từng nhãn $i$, giá trị xác suất dự đoán 
$\hat{p}(\ytilde=i; \vect{x},\vect{\theta})$ đưa ra bởi mô hình $\vect{\theta}$,
nếu không nhỏ chỉ tiêu $t_i$ thì ta cho rằng có khả năng là nhãn $i$ có thể đúng với mẫu $\x$. 
Tập hợp các nhãn $i$'s có thể đúng với mẫu $\x$ là

$\left\{l\in[m]: \hat{p}(\ytilde=l;\x,\vect{\theta})\geq t_l\right\}\neq\emptyset;$ 

Từ tập đó ta chọn nhãn $j$ có xác suất dự đoán lớn nhất để nhận định lớp thật của $\x$ chắc hẳn là $j$. 
"""

# ╔═╡ 0b7ce1c6-4c20-434d-9d05-9546c10acfa5
md"""
### Ma trận tự tin

Gọi $\X_{\ytilde=i,\ystar=j}$ là tập (không tường minh) các mẫu có nhãn quan sát là $i$ và nhãn thật là $j$, ta ước lượng nó như sau bằng cách sử dụng các chỉ tiêu tự tin $t_j$ cho từng lớp $j\in[m]$:
"""

# ╔═╡ d928166c-6892-4326-ada0-6fe984b47a66
md"""
$\hat{\X}_{\ytilde=i,\ystar=j} \coloneqq 
\left\{\x\in\X_{\ytilde=i}:
\mathop{\rm arg max} 
\limits_{l\in[m]: \hat{p}(\ytilde=l;\x,\vect{\theta})\geq t_l}
\hat{p}(\ytilde=l;\x,\vect{\theta}) \equiv j
\right\}\label{eq1}\tag{1}$
"""

# ╔═╡ 14a6267f-2573-4078-945e-224297d49921
md"""
Định nghĩa ma trận tự tin $\vect{C}_{\ytilde,\ystar}$ kích thước $m\times m$
dung nạp kích thước của các tập $\hat{\X}_{\ytilde=i,\ystar=j}$:

$\vect{C}_{\ytilde,\ystar}[i][j] \coloneqq  |\hat{\X}_{\ytilde=i,\ystar=j} |$
"""

# ╔═╡ 52349481-60e8-4fc9-b91c-daef381ec5c1
md"""
### Độ tự tin trung bình

Độ tự tin trung bình trong lớp $i\in[m]$ là

$t_i = \frac{1}{|\X_{\ytilde=i}|} \sum_{\x\in\X_{\ytilde=i}}
\hat{p}(\ytilde=i; \vect{x},\vect{\theta})
\label{eq2}\tag{2}$
"""

# ╔═╡ 81d79128-2d03-484f-ae5f-71449f1f3c64
md"""
### Ước lượng ma trận nhiễu


"""

# ╔═╡ 3def2e10-a802-4973-9799-601ebe91f1eb
md"""
## **Tham khảo**

- [An Introduction to Confident Learning: Finding and Learning with Label Errors in Datasets (curtisnorthcutt.com)](https://l7.curtisnorthcutt.com/confident-learning)
- [cleanlab/cleanlab: The standard data-centric AI package for data quality and machine learning with messy, real-world data and labels. (github.com)](https://github.com/cleanlab/cleanlab)
- [Are Label Errors Imperative? Is Confident Learning Useful? | by Suneeta Mall | May, 2022 | Towards Data Science (medium.com)](https://medium.com/towards-data-science/confident-learning-err-did-you-say-your-data-is-clean-ef2597903328)
"""

# ╔═╡ 58f65407-995c-4095-8958-17668cb7cedf
restroom_icon_url = "https://cdn-icons-png.flaticon.com/512/995/995035.png";

# ╔═╡ ea5ba94f-8061-4d52-9980-e273c022953c
md""" ## Lỗi quan sát nhãn là gì?

Với một đối tượng khảo sát, quan sát viên sẽ quan sát, xem xét, nghiên cứu rồi gán một nhãn nhất định cho dữ liệu đó. Trong môi trường lý tưởng thì ta sẽ nhận định và gán đúng nhãn “chân lý" cho đối tượng.


$(Resource(restroom_icon_url, :width => 200))
[Restroom icons created by Freepik — Flaticon](https://www.flaticon.com/free-icons/restroom)

Ví dụ ta có thể quan sát ngực, bụng, mông của một người nào đó và nhận định giới tính. Trong thực tế thì có thể xảy ra nhầm lẫn ở một bước nào đó trong quá trình từ khi bắt đầu quan sát cho đến khi gắn xong nhãn. Nhầm lẫn đó có thể dẫn tới gán nhầm nhãn “Nam” cho đối tượng vốn là “Nữ”, hoặc ngược lại. Chúng ta gọi "lỗi quan sát nhãn" và "lỗi gắn nhãn" với cùng một ý nghĩa. 

Có thể có một số nam giới và nữ giới có số đo 3 vòng khá giống nhau, nhưng “đương nhiên" họ có 2 giới tính khác nhau, tức là các nhãn “chân lý" của họ là khác nhau về bản chất, chứ không nhất thiết có liên quan đến việc gắn nhãn có lỗi hay không.

Nói cách khác từ số đo 3 vòng ta có thể không suy đoán được chắc chắn 100% nhưng có thể tính được xác suất giới tính Nam/Nữ của đối tượng. Quy tắc hay mô hình suy đoán có thể học được từ một tập dữ liệu có số đo 3 vòng và giới tính tương ứng của nhiều mẫu người khác nhau. Nếu trong tập dữ liệu này có những nhãn giới tính bị gắn sai thì việc học xác suất “chân lý” sẽ bị lệch lạc.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
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

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

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

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

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

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

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
# ╟─fb4735cc-70fb-443b-97de-def822e0c826
# ╟─d83aec0b-4418-425e-a91e-b6d17b1fff69
# ╟─ea5ba94f-8061-4d52-9980-e273c022953c
# ╟─7013e482-ec10-42cc-813f-cd7c29759757
# ╟─dbd6bcfa-fd0b-11ec-3928-79a4a6776291
# ╟─bc77351a-9d49-4fa6-854d-ed7adba9c300
# ╟─b51e3d11-aedc-431d-b591-d808f0a884dc
# ╟─18943cbd-4045-48cc-9857-e42e86aa8556
# ╟─0b7ce1c6-4c20-434d-9d05-9546c10acfa5
# ╟─d928166c-6892-4326-ada0-6fe984b47a66
# ╟─14a6267f-2573-4078-945e-224297d49921
# ╟─52349481-60e8-4fc9-b91c-daef381ec5c1
# ╟─81d79128-2d03-484f-ae5f-71449f1f3c64
# ╟─3def2e10-a802-4973-9799-601ebe91f1eb
# ╟─58f65407-995c-4095-8958-17668cb7cedf
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
