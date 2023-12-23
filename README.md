# CHƯƠNG 1.  TÌM HIỂU SO SÁNH CÁC PHƯƠNG PHÁP OPTIMIZER TRONG HUẤN LUYỆN MÔ HÌNH HỌC MÁY
### 1.1 Optimizer là gì
Optimizer là các thuật toán hoặc phương pháp được sử dụng để giảm thiểu hàm mất mát (loss function). Optimizer là các hàm toán học phụ thuộc vào các tham số có thể học được của mô hình, tức là trọng số (weight) và độ lệch (bias). Optimizer  giúp ta biết cách thay đổi trọng số và tốc độ học (learning rate) của mạng neural để giảm tổn thất.
### 1.2 Các loại optimizer
*1.2.1 Gradient Decent* 

Gradient Descent là thuật toán tối ưu hóa cơ bản nhất nhưng được sử dụng nhiều nhất. Nó được sử dụng rất nhiều trong các thuật toán phân loại và hồi quy tuyến tính. Lan truyền ngược trong mạng nơ-ron cũng sử dụng thuật toán Gradient Descent.
Gradient descent là một thuật toán tối ưu hóa dựa trên một hàm lồi (convex function) và điều chỉnh các tham số của nó một cách lặp lại để giảm thiểu một hàm cho trước đến giá trị tối thiểu cục bộ. Gradient Descent giảm thiểu một hàm mất mát một cách lặp lại bằng cách di chuyển theo hướng ngược với hướng tăng nhanh nhất. Nó phụ thuộc vào đạo hàm của hàm mất mát để tìm giá trị nhỏ nhất. Trong Machine Learning, Gradient Descent sử dụng dữ liệu từ toàn bộ bộ dữ liệu huấn luyện để tính đạo hàm của hàm chi phí đối với các tham số, điều này đòi hỏi một lượng lớn bộ nhớ và làm chậm quá trình.

![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/fb963948-1cd0-445a-abd3-c835299df7ba)
![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/ec2c07bb-4327-44d1-9b6a-10d6b93e2a62)
![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/b3cf3f26-1423-4d87-b07e-f15949b627f0)

*1.2.2 Stochastic Gradient Descent (SGD)*

Giảm dần độ dốc ngẫu nhiên (SGD) là một kỹ thuật tối ưu hóa lặp lại được sử dụng rộng rãi trong học máy và học sâu. Đây là một biến thể của Gradient Descent cung cấp các cập nhật cho các tham số mô hình (trọng số) dựa trên độ dốc của hàm mất được tính toán trên một phần dữ liệu huấn luyện được chọn ngẫu nhiên, thay vì trên tập dữ liệu hoàn chỉnh.
Nguyên tắc cốt lõi của SGD là chọn một phần ngẫu nhiên nhỏ của dữ liệu huấn luyện, được gọi là mini-batch và tính toán độ dốc của hàm mất đối với các tham số mô hình chỉ sử dụng phần đó. Độ dốc này sau đó được sử dụng để cập nhật các tham số. Quy trình được tiếp tục với một mini-batch ngẫu nhiên mới cho đến khi thuật toán hội tụ hoặc đạt được điều kiện dừng xác định trước.	
SGD cung cấp nhiều lợi thế khác nhau so với Gradient Decent truyền thống, chẳng hạn như hội tụ nhanh hơn và nhu cầu bộ nhớ thấp hơn, đặc biệt đối với các bộ dữ liệu lớn. Nó cũng có khả năng phục hồi tốt hơn đối với dữ liệu nhiễu và không cố định, đồng thời có thể thoát khỏi mức tối thiểu cục bộ (local minima). Tuy nhiên, nó có thể cần nhiều lần lặp hơn để hội tụ hơn là Gradient Decent và tốc độ học (learning rate) cần được hiệu chỉnh cẩn thận để đảm bảo sự hội tụ. 
Giảm dần độ dốc ngẫu nhiên là một kỹ thuật tối ưu hóa lặp lại sử dụng các nhóm dữ liệu nhỏ để hình thành kỳ vọng về độ dốc thay vì độ dốc đầy đủ bằng cách sử dụng tất cả dữ liệu có sẵn. Đó là đối với trọng số (W) và hàm mất mát L chúng ta có:

![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/4cc57974-1521-4e38-9ef8-be0fc966607b)

Trong đó n là tốc độ học (learning rate). SGD giảm sự dư thừa so với phương pháp giảm độ dốc hàng loạt - tính toán lại độ dốc cho các ví dụ tương tự trước mỗi lần cập nhật tham số - vì vậy nó thường nhanh hơn nhiều.

*1.2.3 Momentum*

Momentum là một chiến lược tối ưu hóa được sử dụng trong học máy và học sâu để đẩy nhanh quá trình đào tạo mạng nơ-ron. Nó dựa trên khái niệm thêm một phần của bản cập nhật trước đó vào bản cập nhật trọng số hiện tại trong suốt quá trình tối ưu hóa. 
Trong tối ưu hóa momentum, độ dốc của hàm chi phí được tính theo từng trọng số trong mạng nơ-ron. Thay vì cập nhật trọng số trực tiếp dựa trên gradient, tối ưu hóa động lượng đưa ra một biến mới, gọi là số hạng động lượng (momentum term), được sử dụng để cập nhật trọng số. Thuật ngữ động lượng là trung bình động của các gradient và nó tích lũy các gradient trước đó để giúp tác động đến hướng tìm kiếm.
Thuật ngữ động lượng có thể được xem là vận tốc của bộ tối ưu hóa. Bộ tối ưu hóa thu được động lượng khi nó đi xuống dốc và dùng để làm giảm các dao động trong quá trình tối ưu hóa. Điều này có thể cho phép trình tối ưu hóa hội tụ nhanh hơn và đạt được mức tối thiểu cục bộ (local minimum) tốt hơn.
Tối ưu hóa động lượng đặc biệt có lợi trong các trường hợp khi bối cảnh tối ưu hóa nhiễu hoặc khi độ dốc thay đổi nhanh. Nó cũng có thể giúp làm trơn tru quá trình tối ưu hóa và tránh trình tối ưu hóa bị mắc kẹt trong mức tối thiểu cục bộ.
Cách mà momentum hoạt động:
Động lực tối ưu hóa giống như một quả bóng lăn xuống dốc. Trong khi độ dốc giảm dần cập nhật các tham số dựa trên độ dốc hiện tại, động lượng sẽ thêm một phần của bản cập nhật trước đó vào phần hiện tại. Điều này giúp trình tối ưu hóa duy trì theo cùng một hướng, giảm dao động và tránh bị kẹt ở cực tiểu cục bộ, giải quyết các nhược điểm của phương pháp giảm độ dốc truyền thống. Quy tắc cập nhật động lượng có thể được viết như sau:

![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/323d5c17-62c2-4249-9cb3-34d62af24e53)

*J(θ) là hàm chi phí..
*∇_θ  J(θ) là độ dốc của hàm chi phí đối với các tham số θ 
*β là số hạng động lượng (momentum term)
*v là vectơ vận tốc hoặc động lượng
*α là tốc độ học

Thông thường, hệ số động lượng được đặt ở mức 0,9. Trình tối ưu hóa tính toán độ dốc của hàm chi phí tại mỗi lần lặp và cập nhật số hạng động lượng dưới dạng trung bình động có trọng số theo cấp số nhân của các độ dốc trước đó. Các tham số sau đó được cập nhật bằng cách loại bỏ số hạng động lượng nhân với tốc độ học.
Nhìn chung, động lượng là một chiến lược tối ưu hóa mạnh mẽ có thể hỗ trợ đẩy nhanh quá trình đào tạo mạng lưới nơ-ron sâu và tăng hiệu suất của chúng.

*1.2.4 AdaGrad(Adaptive Gradient Descent)*

Adagrad (Adaptive Gradient Descent) là một kỹ thuật tối ưu hóa được sử dụng trong học máy và học sâu để tối ưu hóa việc đào tạo mạng nơ-ron.
Phương pháp Adagrad điều chỉnh tốc độ học của từng tham số của mạng nơ-ron một cách thích ứng trong quá trình huấn luyện. Cụ thể, nó điều chỉnh tốc độ học của từng tham số dựa trên độ dốc thu được trước đó cho tham số đó. Nói cách khác, các tham số có độ dốc lớn sẽ có tốc độ học thấp hơn, trong khi những tham số có độ dốc nhỏ sẽ có tốc độ học lớn hơn. Điều này giúp ngăn tốc độ học giảm xuống quá nhanh đối với các tham số thường xuyên xảy ra và cho phép hội tụ quá trình đào tạo nhanh hơn.
Kỹ thuật Adagrad đặc biệt hiệu quả trong việc xử lý dữ liệu thưa thớt, khi các phần của đặc tính đầu vào có tần số thấp hoặc không có. Trong những trường hợp này, Adagrad có thể thay đổi tốc độ học của từng tham số một cách thích ứng, cho phép xử lý dữ liệu thưa thớt tốt hơn. Nhìn chung, Adagrad là một phương pháp tối ưu hóa mạnh mẽ có thể hỗ trợ tăng tốc độ đào tạo mạng lưới thần kinh sâu và nâng cao hiệu suất của chúng.
Cách thức hoạt động: Nguyên tắc chính của AdaGrad là chia tỷ lệ learning rate cho từng tham số theo tổng độ dốc bình phương quan sát được trong quá trình đào tạo. Các bước của thuật toán như sau:
*	Khởi tạo biến
Khởi tạo các tham số θ và một hằng số nhỏ ϵ để tránh chia cho 0.
Khởi tạo tổng của biến gradient bình phương G bằng các số 0, có hình dạng giống như θ.
*	Tính toán độ dốc
	Tính gradient của hàm mất mát đối với từng tham số, ∇θJ(θ)
*	Tích lũy gradient bình phương
	Cập nhật tổng gradient bình phương G cho mỗi tham số i: 
G[i] += (∇θJ(θ[i]))²
*	Cập nhật tham số
	Cập nhật từng tham số bằng tốc độ học thích ứng: 
θ[i] -= (η / (√(G[i]) + ϵ)) * ∇θJ(θ[i])
η :tốc độ học
∇θJ(θ[i]) : gradient của hàm mất mát đối với tham số θ[i]. 

*1.2.5 AdaDelta*

Adadelta là một kỹ thuật tối ưu hóa được sử dụng trong học máy và học sâu để tối ưu hóa việc đào tạo mạng lưới thần kinh. Đây là một biến thể của phương pháp Adagrad khắc phục được một số nhược điểm của nó.
Phương pháp Adadelta sửa đổi tốc độ học của từng tham số theo cách tương tự như Adagrad, nhưng thay vì giữ lại tất cả các gradient trước đó, nó chỉ giữ lại mức trung bình động của các gradient bình phương. Điều này giúp giảm nhu cầu bộ nhớ của phương pháp.
Ngoài ra, Adadelta sử dụng một phương pháp gọi là "cập nhật delta" để thay đổi tốc độ học. Thay vì sử dụng tốc độ học tập đã đặt, Adadelta sử dụng tỷ lệ bình phương trung bình gốc (RMS) của các gradient trước đó và RMS của các bản cập nhật trước đây để mở rộng tốc độ học tập. Điều này giúp ngăn chặn hơn nữa tốc độ học tập giảm quá sớm đối với các đặc điểm thường xuyên xảy ra.
Giống như Adagrad, Adadelta đặc biệt hiệu quả trong việc xử lý dữ liệu thưa thớt, nhưng nó cũng có thể hoạt động tốt hơn trong trường hợp Adagrad có thể hội tụ quá nhanh. 
AdaDelta là một kỹ thuật tối ưu hóa ngẫu nhiên cho phép áp dụng phương pháp tốc độ học theo chiều cho SGD. Nó là một phần mở rộng của Adagrad nhằm tìm cách giảm tốc độ học tập giảm dần và đơn điệu. Thay vì tích lũy tất cả các gradient bình phương trong quá khứ, Adadelta giới hạn cửa sổ của các gradient trong quá khứ được tích lũy ở một kích thước cố định w.
Thay vì lưu trữ không hiệu quả w gradient bình phương trước đó, tổng gradient được xác định đệ quy là trung bình phân rã của tất cả gradient bình phương trước đó. Khi đó, mức trung bình đang chạy ở bước thời gian chỉ phụ thuộc vào độ dốc trung bình trước đó và hiện tại:

![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/ae20e9c6-7b0d-459b-9113-f5b8977493fc)

*E〖[g^2]〗_t: Trung bình chạy của gradient bình phương tại thời điểm t.
*γ: Hệ số suy giảm hoặc hệ số làm mịn, giá trị từ 0 đến 1 xác định trọng số cho giá trị trung bình chạy trước đó.
*γE〖[g^2]〗_(t-1 ): Trung bình chạy của các gradient bình phương ở bước thời gian trước đó (t−1).
*g_t: Độ dốc của hàm chi phí đối với các tham số tại thời điểm t.

Thông thường γ  được đặt ở khoảng 0,9. Viết lại cập nhật SGD theo vectơ cập nhật tham số:
![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/2e5d2258-fdf7-4dd8-8c1d-5389b78770d4)

AdaDelta có dạng:

![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/428c3f4d-d143-428c-82b4-7c90f04a88cc)

*∆θ_t: Cập nhật tham số tại thời điểm t.
*n: Tốc độ học, một đại lượng vô hướng dương xác định kích thước bước trong không gian tham số.
*g_t: Gradient của hàm chi phí đối với các tham số tại thời điểm t
*E[g^2 ]_t: Trung bình chạy của gradient bình phương tại thời điểm t
*ϵ: Hằng số nhỏ được thêm vào để ổn định số học nhằm tránh chia cho 0 ở mẫu số.
Ưu điểm chính của AdaDelta là chúng ta không cần đặt tốc độ học mặc định.
Nhìn chung, Adadelta là một kỹ thuật tối ưu hóa mạnh mẽ có thể hỗ trợ đẩy nhanh quá trình đào tạo mạng lưới thần kinh sâu và tăng hiệu suất của chúng, đồng thời giải quyết một số nhược điểm của Adagrad.

*1.2.6 RMS-Prop (Root Mean Square Propagation)*

RMSProp (Root Mean Square Propagation) là một phương pháp tối ưu hóa được sử dụng trong học máy và học sâu để tối ưu hóa việc đào tạo mạng lưới thần kinh.
Giống như Adagrad và Adadelta, RMSProp sửa đổi tốc độ học của từng tham số trong suốt quá trình đào tạo. Tuy nhiên, thay vì thu thập tất cả các gradient trước đó như Adagrad, RMSProp tính toán mức trung bình động của các gradient bình phương. Điều này giúp thuật toán sửa đổi tốc độ học chậm hơn và ngăn tốc độ học giảm xuống quá sớm.
Kỹ thuật RMSProp cũng sử dụng hệ số phân rã để hạn chế ảnh hưởng của độ dốc trước đó đến tốc độ học. Hệ số phân rã này cho phép thuật toán tăng trọng số cho các gradient gần đây và ít trọng số hơn cho các gradient cũ hơn.
Một trong những ưu điểm chính của RMSProp so với Adagrad là nó có thể xử lý các mục tiêu không cố định, trong đó chức năng cơ bản mà mạng nơ-ron đang cố gắng bắt chước thay đổi theo thời gian. Trong một số trường hợp nhất định, Adagrad có thể hội tụ quá nhanh, trong khi RMSProp có thể điều chỉnh tốc độ học theo hàm mục tiêu thay đổi.
Đây là cách RMSProp hoạt động:
*	Khởi tạo: Khởi tạo một biến trung bình đang chạy về 0, trong đó là độ dốc của hàm chi phí đối với các tham số.
*	Tính toán độ dốc: Tại mỗi lần lặp, tính toán độ dốc của hàm chi phí đối với các tham số.
*	Cập nhật trung bình chạy:
Cập nhật giá trị trung bình đang chạy bằng cách sử dụng phân rã theo cấp số nhân:
![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/68b4d6b4-17de-47e9-8cbe-8aba02ae185e)

Ở đây γ là hệ số phân rã (thường gần bằng 1) xác định trọng số của đường trung bình chạy trước đó so với độ dốc hiện tại.
*Cập nhật thông số:
![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/14ef2a3b-61dd-4f25-a939-8828c7d37cc7)

*1.2.7 Adam(Adaptive Moment Estimation)*

Adam (Adaptive Moment Estimation) là một kỹ thuật tối ưu hóa được sử dụng trong học máy và học sâu để tối ưu hóa việc đào tạo mạng lưới thần kinh.

Adam tích hợp các khái niệm về động lượng và RMSProp. Nó duy trì mức trung bình động của khoảnh khắc thứ nhất và thứ hai của gradient, tương ứng là giá trị trung bình và phương sai của gradient. Đường trung bình động của thời điểm ban đầu, có thể so sánh với số hạng động lượng trong các phương pháp tối ưu hóa khác, hỗ trợ trình tối ưu hóa tiếp tục tiến triển theo cùng một hướng ngay cả khi độ dốc nhỏ hơn. Trung bình động của thời điểm thứ hai, giống hệt với thuật ngữ RMSProp, hỗ trợ trình tối ưu hóa điều chỉnh tốc độ học cho từng tham số dựa trên phương sai của độ dốc.

Adam cũng bao gồm một giai đoạn hiệu chỉnh độ lệch để thay đổi các đường trung bình động vì chúng có độ lệch về 0 khi bắt đầu quá trình tối ưu hóa. Điều này giúp tăng hiệu suất của thuật toán tối ưu hóa trong giai đoạn đầu đào tạo.

Adam là một kỹ thuật tối ưu hóa phổ biến vì khả năng hội tụ nhanh chóng và quản lý độ dốc nhiễu hoặc thưa thớt. Ngoài ra, nó không yêu cầu cài đặt thủ công các siêu tham số như suy giảm tốc độ học tập hoặc hệ số động lượng, giúp sử dụng dễ dàng hơn các kỹ thuật tối ưu hóa khác.

Cách hoạt động:
*Khởi tạo:
Khởi tạo các tham số (ước tính thời điểm ban đầu), (ước tính thời điểm thô thứ hai ban đầu).
*Đặt t = 0 (bộ đếm lặp)
Chọn siêu tham số: α (tốc độ học), β_(1 )(tốc độ phân rã theo cấp số nhân cho ước tính khoảnh khắc đầu tiên), β_(2 ) tốc độ phân rã theo cấp số nhân cho ước tính mô men thô thứ hai), ϵ (hằng số nhỏ để ổn định số).
*Tính toán độ dốc:
Tại mỗi lần lặp t, hãy tính độ dốc của hàm chi phí đối với các tham số.
*Cập nhật ước tính thời điểm đầu tiên:
Cập nhật ước tính thời điểm đầu tiên (momentum term):
 ![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/d3de1e7f-9e77-4eb6-ab9b-0fa94cff6edf)

*Cập nhật ước tính khoảnh khắc thô thứ hai:
Cập nhật ước lượng mômen thô thứ hai:
![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/acfc18ef-99e2-43f6-a3f5-d51749493a75)

*Hiệu chỉnh sai lệch:
Sửa sai lệch trong ước tính thời điểm thứ nhất và thứ hai, có xu hướng sai lệch về 0 trong các lần lặp đầu tiên:
![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/c4b374a3-690d-482e-a2e8-3ccf1e2c79e5)

*Cập nhật tham số:

![image](https://github.com/loiloi26/CuoiKi_MachineLearning/assets/94375939/64f9fa1a-050c-4011-8411-9f1e29b73979)

Ở đây θ thể hiện các thông số đang được tối ưu hóa.
*Lặp lại:
Tăng bộ đếm lần lặp t và lặp lại các bước 2-6 cho đến khi hội tụ hoặc số lần lặp xác định.
Nhìn chung, Adam là một phương pháp tối ưu hóa mạnh mẽ có thể giúp đẩy nhanh quá trình đào tạo mạng lưới thần kinh sâu và tăng hiệu suất của chúng.

*1.3 Phân tích so sánh*

|     Optimizer    | Ưu điểm                                                      | Nhược điểm                                                                                                                                                                                                                                                                               |
|:----------------:|--------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Gradient Descent |      - Tính toán dễ dàng. - Dễ để thực hiện.  - Dễ hiểu.     |      - Có thể mắc kẹt ở mức tối thiểu cục bộ. - Trọng số được thay đổi sau khi tính toán độ dốc trên toàn bộ tập dữ liệu. Vì vậy, nếu    tập dữ liệu quá lớn thì có thể mất nhiều năm để hội tụ về mức tối thiểu. - Yêu cầu bộ nhớ lớn để tính toán độ dốc trên toàn bộ tập dữ liệu.     |
|                  |                                                              |                                                                                                                                                                                                                                                                                          |
|                  |                                                              |                                                                                                                                                                                                                                                                                          |
|                  |                                                              |                                                                                                                                                                                                                                                                                          |
|                  |                                                              |                                                                                                                                                                                                                                                                                          |
|                  |                                                              |                                                                                                                                                                                                                                                                                          |





