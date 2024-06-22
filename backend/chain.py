import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence

# import weaviate
from constants import WEAVIATE_DOCS_INDEX_NAME
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ingest import get_embeddings_model
from langchain_community.chat_models import ChatCohere
from langchain_community.vectorstores import Weaviate
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
    chain,
)

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langsmith import Client

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


import re
import string
import json


dict_replace = {
    "2k|2k|2k|2k|2k": "200",
    "a": "anh",
    "ac": "anh chị",
    "a/c": "anh chị",
    "add|ad|adm|addd": "admin",
    "ak|ah": "à",
    "b": "bạn",
    "baoh|bh": "bao giờ",
    "bhyt": "bảo hiểm y tế",
    "bn|bnh|baonh": "bao nhiêu",
    "bâyh": "bây giờ",
    "cccd": "căn cước công dân",
    "chòa": "chào",
    "cj": "chị",
    "ck": "chuyển khoản",
    "chỉ tiêu": "chỉ tiêu tuyển sinh",
    "chỉ tiêu tuyển sinh tuyển sinh": "chỉ tiêu tuyển sinh",
    "clc": "chất lượng cao",
    "cmnd": "chứng minh nhân dân",
    "dgnl": "đánh giá năng lực",
    "dgtd": "đánh giá tư duy",
    "dhqg": "đại học quốc gia",
    "dkxt": "đăng ký xét tuyển",
    "dko": "được không",
    "đko": "được không",
    "dky|dki|đăng kí|dki|đki": "đăng ký",
    "đkxt|dkxt": "đăng ký xét tuyển",
    "hvcnbtvt": "Học viện Công nghệ Bưu chính Viễn thông",
    "d|đ": "điểm",
    "e": "em",
    "gddt|gd dt|gdđt|gd đt": "giáo dục đào tạo",
    "gd": "giáo dục",
    "hc": "học",
    "hcm": "hồ chí minh",
    "hnay": "hôm nay",
    "hqua": "hôm qua",
    "điểm tuyển sinh": "điểm trúng tuyển",
    "điểm đỗ": "điểm trúng tuyển",
    "điểm chuẩn|điểm": "điểm trúng tuyển",
    "điểm trúng tuyển đỗ": "điểm trúng tuyển",
    "điểm trúng tuyển điểm trúng tuyển": "điểm trúng tuyển",
    "điểm trúng tuyển trúng tuyển": "điểm trúng tuyển",
    "điểm trúng tuyển tuyển sinh": "điểm trúng tuyển",
    "điểm trúng tuyển": "điểm trúng tuyển",
    "điểm trúng tuyển nổi bật": "điểm nổi bật",
    "điểm trúng tuyển ưu tiên": "điểm ưu tiên",
    "điểm trúng tuyển cộng": "điểm cộng",
    "cộng điểm trúng tuyển": "cộng điểm",
    "điểm trúng tuyển thưởng": "điểm thưởng",
    "hv": "học viện",
    "j": "gì",
    "ko|k|kh|khg|kg|hong|hok|khum": "không",
    "ktx": "ký túc xá",
    "kt": "Kế toán",
    "kv|khvuc|kvuc": "khu vực",
    "lm": "làm",
    "lpxt": "lệ phí xét tuyển",
    "m": "mình",
    "mn": "mọi người",
    "tkdh": "thiết kế đồ họa",
    "mk": "mình",
    "nnao": "như nào",
    "ntn": "như thế nào",
    "nv|nvong": "nguyện vọng",
    "oke|oki|okeee": "ok",
    "p": "phải",
    "pk": "phải không",
    "q1|q9": "quận",
    "r": "rồi",
    "sdt|sđt|dt|đt": "số điện thoại",
    "sv": "sinh viên",
    "t": "tôi",
    "thpt": "trung học phổ thông",
    "thptqg": "trung học phổ thông quốc gia",
    "tk": "tài khoản",
    "tmdt": "Thương mại điện tử",
    "tnthpt": "trắc nghiệm trung học phổ thông",
    "tp hcm": "thành phố hồ chí minh",
    "trg|tr|trườg": "trường",
    "trường": "học viện",
    "ttnv": "thứ thự nguyện vọng",
    "tt": "thông tin",
    "uh|uk|um": "ừ",
    "vc": "việc",
    "vd": "ví dụ",
    "vs": "với",
    "z|v": "vậy",
    "đc|dc": "được",
    "năm nay": "2024",
    "năm ngoái|năm trước": "2023",
    "A1|a1": "A01",
    "A0|a0": "A00",
    "D1|d1": "D01",
    "A01": "tổ hợp xét tuyển A01",
    "A00": "tổ hợp xét tuyển A00",
    "D01": "tổ hợp xét tuyển D01",
    "hsg": "học sinh giỏi",
    "giải 1": "giải nhất học sinh giỏi",
    "giải nhất học sinh giỏi học sinh giỏi": "giải nhất học sinh giỏi",
    "giải 2": "giải nhì học sinh giỏi",
    "giải nhì học sinh giỏi học sinh giỏi": "giải nhì học sinh giỏi",
    "giải 3": "giải ba học sinh giỏi",
    "giải ba học sinh giỏi học sinh giỏi": "giải ba học sinh giỏi",
    "tổ hợp": "tổ hợp xét tuyển",
    "khối": "tổ hợp xét tuyển",
    "tổ hợp xét tuyển tổ hợp xét tuyển": "tổ hợp xét tuyển",
    "hsnl": "hồ sơ năng lực",
    "hthuc": "hình thức",
    "hthức": "hình thức",
    "hso": "hồ sơ",
    "hs": "hồ sơ",
    "nl": "năng lực",
    "hssv": "hồ sơ sinh viên",
    "xttn": "xét tuyển tài năng",
    "xtkh|xthk": "xét tuyển kết hợp",
    "xt": "xét tuyển",
    "xtkn": "xét tuyển kết hợp",
    "hba": "học bạ",
    "rv": "review",
    "review": "thông tin",
    "pthuc": "phương thức",
    "pt": "phương thức",
    "ielts": "IELTS",
    "toefl": "TOEFL",
    "toeic": "TOEIC",
    "sat": "SAT",
    "act": "ACT",
    "toefl itp": "TOEFL ITP",
    "toefl ibt": "TOEFL iBT",
    "xét": "xét tuyển",
    "xét tuyển tuyển": "xét tuyển",
    "Tổ hợp xét tuyển": "tổ hợp xét tuyển",
    "ktdtvt": "kỹ thuật điện tử viễn thông",
    "dtvt|dt vt|đtvt|đt vt": "điện tử viễn thông",
    "vt": "Viễn thông",
    "attt": "An toàn thông tin",
    "ttdpt|ttđpt": "Truyền thông đa phương tiện",
    "cn dpt|cndpt": "Công nghệ đa phương tiện",
    "udu": "Công nghệ thông tin định hướng ứng dụng",
    "cntt udu": "Công nghệ thông tin định hướng ứng dụng",
    "cntt ứng dụng": "Công nghệ thông tin định hướng ứng dụng",
    "công nghệ thông tin ứng dụng": "Công nghệ thông tin định hướng ứng dụng",
    "cntt": "Công nghệ thông tin",
    "dpt|đpt": "đa phương tiện",
    "khmt": "Khoa học máy tính",
    "ktdk|ktđk": "Kỹ thuật điều khiển",
    "ktdktdh": "Kỹ thuật điều khiển và tự động hóa",
    "iot": "Công nghệ Internet vạn vật (IoT)",
    "qtkd": "Quản trị kinh doanh",
    "ngành kt": "Ngành kế toán",
    "mmt": "Mạng máy tính",
    "bc": "Báo chí",
    "cntc": "Công nghệ tài chính (Fintech)",
    "marketting|makettting|mkt": "marketing",
}


data_map_key = [
    {
        "keys": [
            "cộng điểm",
            "điểm cộng",
            "điểm ưu tiên",
            "ưu tiên cộng",
            "điểm thưởng",
        ],
        "insert_key": [],
        "requirements": [
            "phương thức xét tuyển, điều kiện xét tuyển, quy trình đăng ký"
        ],
        "level": 1,
        "document": """Đối với thí sinh có giải học sinh giỏi có thể xét tuyển Phương thức 1 - xét tuyển tài năng hoặc xét tuyển Phương thức 3 - xét tuyển kết hợp hoặc xét tuyển Phương thức 4 - xét tuyển dựa vào kết quả bài thi đánh giá năng lực, đánh giá tư duy.
Thí sinh có giải học sinh giỏi được cộng điểm thành tích nếu xét tuyển phương thức 1 với điều kiện về giải Nhất, Nhì, Ba, KK HSG tỉnh, TP các môn Toán, Lý, Hóa, Tin và có kết quả điểm trung bình chung học tập lớp 10, 11, 12 hoặc học kỳ 1 lớp 12 đạt từ 7,5 trở lên và có hạnh kiểm Khá trở lên
Thí sinh có giải học sinh giỏi cộng điểm thưởng, cộng điểm ưu tiên nếu xét tuyển phương thức 3 với điều kiện kết quả điểm trung bình chung học tập từ lớp 10 đến lớp 12 hoặc học kỳ 1 lớp 12 (nếu chưa có kết quả năm học lớp 12) đạt từ 7,5 trở lên hoặc xét tuyển phương thức 4
Chi tiết cộng điểm như sau:
-  có giải khuyến khích học sinh giỏi quốc gia được cộng 40 điểm thành tích Phương thức 1 hoặc 3 điểm ưu tiên Phương thức 3, 4
-  có giải nhất học sinh giỏi cấp tỉnh/thành phố trung ương được cộng 35 điểm thành tích Phương thức 1 hoặc 2.5 điểm ưu tiên Phương thức 3, 4
-  có giải nhì học sinh giỏi cấp tỉnh/thành phố trung ương được cộng 30 điểm thành tích Phương thức 1 hoặc 2 điểm ưu tiên Phương thức 3, 4
-  có giải ba học sinh giỏi cấp tỉnh/thành phố trung ương được cộng 25 điểm thành tích Phương thức 1 hoặc 1.5 điểm ưu tiên Phương thức 3, 4
-  có giải khuyến khích học sinh giỏi cấp tỉnh/thành phố trung ương được cộng 20 điểm thành tích Phương thức 1 hoặc 1 điểm ưu tiên Phương thức 3, 4
-  học sinh chuyên không có giải được cộng 25 điểm thành tích
Ghi chú: Thí sinh chỉ được lựa chọn một 01 loại thành tích cao nhất.""",
        "answer": [],
    },
    {
        "keys": [
            "tổ hợp xét tuyển",
            "tổ hợp",
        ],
        "insert_key": [],
        "requirements": [""],
        "level": 1,
        "document": """"Tổ hợp xét tuyển\n- Tổ hợp A00: Toán, Vật lý, Hóa học.\n- Tổ hợp A01: Toán, Vật lý, Tiếng Anh.\n- Tổ hợp D01: Toán, Văn, Tiếng Anh.\n\ntổ hợp xét tuyển các ngành của học viện gồm 20 ngành sau:\n1. ngành Công nghệ thông tin, mã ngành 7480201, tổ hợp xét tuyển A00, A01\n2. ngành Cử nhân Công nghệ thông tin định hướng ứng dụng, mã ngành 7480201_UDU, tổ hợp xét tuyển A00, A01, đào tạo tại phía Bắc\n3. ngành Mạng máy tính và truyền thông dữ liệu (CT Kỹ thuật dữ liệu), mã ngành 7480102, tổ hợp xét tuyển A00, A01, đào tạo tại phía Bắc\n4. ngành Khoa học máy tính (định hướng Khoa học dữ liệu), mã ngành 7480101, tổ hợp xét tuyển A00, A01, đào tạo tại phía Bắc\n5. ngành Kỹ thuật Điện tử viễn thông, mã ngành 7520207, tổ hợp xét tuyển A00, A01\n6. ngành Công nghệ kỹ thuật Điện, điện tử, mã ngành 7510301, tổ hợp xét tuyển A00, A01\n7. ngành Kỹ thuật Điều khiển và tự động hóa, mã ngành 7520216, tổ hợp xét tuyển A00, A01\n8. ngành Công nghệ Inernet vạn vật (IoT), mã ngành 7520208, tổ hợp xét tuyển A00, A01, đào tạo tại phía Nam\n9. ngành An toàn thông tin, mã ngành 7480202, tổ hợp xét tuyển A00, A01\n10. ngành Công nghệ đa phương tiện, mã ngành 7329001, tổ hợp xét tuyển A00, A01, D01\n11. ngành Truyền thông đa phương tiện, mã ngành 7320104, tổ hợp xét tuyển A00, A01, D01, đào tạo tại phía Bắc\n12. ngành Báo chí, mã ngành 7320101, tổ hợp xét tuyển A00, A01, D01, đào tạo tại phía Bắc\n13. ngành Quản trị kinh doanh, mã ngành 7340101, tổ hợp xét tuyển A00, A01, D01\n14. ngành Thương mại điện tử, mã ngành 7340122, tổ hợp xét tuyển A00, A01, D01, đào tạo tại phía Bắc\n15. ngành Marketing, mã ngành 7340115, tổ hợp xét tuyển A00, A01, D01\n16. ngành Kế toán, mã ngành 7340301, tổ hợp xét tuyển A00, A01, D01\n17. ngành Công nghệ tài chính (Fintech), mã ngành 7340205, tổ hợp xét tuyển A00, A01, D01, đào tạo tại phía Bắc\n18. ngành Công nghệ thông tin_CLC, mã ngành 7480201_CLC, tổ hợp xét tuyển A00, A01\n19. ngành Marketing_CLC, mã ngành 7340115_CLC, tổ hợp xét tuyển A00, A01, D01\n20. ngành Kế toán_CLC (chuẩn quốc tế ACCA), mã ngành 7340301_CLC, tổ hợp xét tuyển A00, A01, D01, đào tạo tại phía Bắc""",
        "answer": [],
    },
    {
        "keys": [
            "quy đổi điểm chứng chỉ ielts",
            "quy đổi điểm chứng chỉ toefl",
            "quy đổi điểm chứng chỉ toefl itp",
            "quy đổi điểm chứng chỉ toefl ibt",
            "quy đổi điểm chứng chỉ tiếng anh",
            "quy đổi điểm",
            "điểm ielts",
            "điểm toefl",
            "điểm toefl itp",
            "điểm toefl ibt",
            "điểm chứng chỉ",
            "quy đổi",
            "đổi điểm",
        ],
        "insert_key": [
            "quy đổi điểm chứng chỉ ielts",
            "quy đổi điểm chứng chỉ toefl",
            "quy đổi điểm chứng chỉ toefl itp",
            "quy đổi điểm chứng chỉ toefl ibt",
            "quy đổi điểm chứng chỉ tiếng anh",
            "quy đổi điểm chứng chỉ tiếng anh",
            "quy đổi điểm chứng chỉ ielts",
            "quy đổi điểm chứng chỉ toefl",
            "quy đổi điểm chứng chỉ toefl itp",
            "quy đổi điểm chứng chỉ toefl ibt",
            "quy đổi điểm chứng chỉ tiếng anh",
            "quy đổi điểm chứng chỉ tiếng anh",
            "quy đổi điểm chứng chỉ tiếng anh",
        ],
        "requirements": [
            "quy đổi điểm chứng chỉ tiếng anh quốc tế",
        ],
        "level": 1,
        "document": """Quy đổi điểm Chứng chỉ tiếng Anh quốc tế
        Đối với Phương thức xét tuyển kết hợp, thí sinh được phép quy đổi điểm môn tiếng Anh trong tổ hợp xét tuyển khi có Chứng chỉ tiếng Anh quốc tế. Cụ thể:
        Quy đổi điểm Chứng chỉ tiếng Anh quốc tế IELTS
        • 7.5 - 9.0 tương đương điểm quy đổi 10 điểm
        • 7.0 tương đương điểm quy đổi 9.5 điểm
        • 6.5 tương đương điểm quy đổi 9.0 điểm
        • 6.0 tương đương điểm quy đổi 8.5 điểm
        • 5.5 tương đương điểm quy đổi 8.0 điểm
        Quy đổi điểm Chứng chỉ tiếng Anh quốc tế TOEFL iBT
        • Từ 102 điểm trở lên tương đương điểm quy đổi 10 điểm
        • 90 - 101 tương đương điểm quy đổi 9.5 điểm
        • 79 - 89 tương đương điểm quy đổi 9.0 điểm
        • 72 – 78 tương đương điểm quy đổi 8.5 điểm
        • 61 - 71 tương đương điểm quy đổi 8.0 điểm
        Quy đổi điểm Chứng chỉ tiếng Anh quốc tế TOEFL ITP
        • Từ 627 điểm trở lên tương đương điểm quy đổi 10 điểm
        • 590 - 626 tương đương điểm quy đổi 9.5 điểm
        • 561 - 589 tương đương điểm quy đổi 9.0 điểm
        • 543 - 560 tương đương điểm quy đổi 8.5 điểm
        • 500 - 542 tương đương điểm quy đổi 8.0 điểm""",
        "answer": [],
    },
    {
        "keys": [
            "chỉ tiêu tuyển sinh",
            "chỉ tiêu",
        ],
        "insert_key": [
            "chỉ tiêu tuyển sinh",
            "chỉ tiêu tuyển sinh",
        ],
        "level": 2,
        "document": """
        Tổng chỉ tiêu tuyển sinh 2024 là 5.060 cho cả 2 cơ sở đào tạo tại Hà Nội và Tp. HCM, trong đó chỉ tiêu tuyển sinh của 02 Cơ sở đào tạo như sau:
        Chỉ tiêu tuyển sinh tại cơ sở phía Bắc (Mã trường: BVH) tổng chỉ tiêu 3900 gồm 420 chỉ tiêu xét tuyển tài năng, 1950 chỉ tiêu thi THPT (Trung học phổ thông), 965 chỉ tiêu xét tuyển kết hợp (XTKH), 565 chỉ tiêu đánh giá năng lực, đánh giá tư duy (ĐGNL, ĐGTD).
        Chỉ tiêu tuyển sinh tại cơ sở phía Nam (Mã trường: BVS) tổng chỉ tiêu 1160 gồm 120 chỉ tiêu xét tuyển tài năng, 575 chỉ tiêu thi THPT (Trung học phổ thông), 280 chỉ tiêu xét tuyển kết hợp (XTKH), 185 chỉ tiêu đánh giá năng lực, đánh giá tư duy (ĐGNL, ĐGTD).

        Chi tiết chỉ tiêu tuyển sinh các ngành/nhóm ngành tại các cơ sở theo từng phương thức xét tuyển như sau:
        chỉ tiêu tuyển sinh ngành Kỹ thuật Điện tử viễn thông tại cơ sở phía Bắc (BVH) (Mã ngành 7520207) là 390 gồm 45 chỉ tiêu xét tuyển tài năng, 195 chỉ tiêu thi THPT, 95 chỉ tiêu xét tuyển kết hợp, 55 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Kỹ thuật Điện tử viễn thông tại cơ sở phía Nam (BVS) (Mã ngành 7520207) là 100 gồm 10 chỉ tiêu xét tuyển tài năng, 50 chỉ tiêu thi THPT, 25 chỉ tiêu xét tuyển kết hợp, 15 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Kỹ thuật Điều khiển và tự động hóa tại cơ sở phía Bắc (BVH) (Mã ngành 7520216) là 80 gồm 10 chỉ tiêu xét tuyển tài năng, 40 chỉ tiêu thi THPT, 20 chỉ tiêu xét tuyển kết hợp, 10 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Kỹ thuật Điều khiển và tự động hóa tại cơ sở phía Nam (BVS) (Mã ngành 7520216) là 85 gồm 10 chỉ tiêu xét tuyển tài năng, 40 chỉ tiêu thi THPT, 20 chỉ tiêu xét tuyển kết hợp, 15 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Công nghệ Kỹ thuật Điện-điện tử tại cơ sở phía Bắc (BVH) (Mã ngành 7510301) là 240 gồm 25 chỉ tiêu xét tuyển tài năng, 120 chỉ tiêu thi THPT, 60 chỉ tiêu xét tuyển kết hợp, 35 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Công nghệ Kỹ thuật Điện-điện tử tại cơ sở phía Nam (BVS) (Mã ngành 7510301) là 90 gồm 10 chỉ tiêu xét tuyển tài năng, 45 chỉ tiêu thi THPT, 20 chỉ tiêu xét tuyển kết hợp, 15 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Công nghệ thông tin tại cơ sở phía Bắc (BVH) (Mã ngành 7480201) là 600 gồm 60 chỉ tiêu xét tuyển tài năng, 300 chỉ tiêu thi THPT, 150 chỉ tiêu xét tuyển kết hợp, 90 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Công nghệ thông tin tại cơ sở phía Nam (BVS) (Mã ngành 7480201) là 180 gồm 15 chỉ tiêu xét tuyển tài năng, 90 chỉ tiêu thi THPT, 45 chỉ tiêu xét tuyển kết hợp, 30 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Cử nhân Công nghệ thông tin định hướng ứng dụng tại cơ sở phía Bắc (BVH) (Mã ngành 7480201_UDU) là 280 gồm 30 chỉ tiêu xét tuyển tài năng, 140 chỉ tiêu thi THPT, 70 chỉ tiêu xét tuyển kết hợp, 40 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Công nghệ thông tin chất lượng cao tại cơ sở phía Bắc (BVH) (Mã ngành 7480201_CLC) là 280 gồm 30 chỉ tiêu xét tuyển tài năng, 140 chỉ tiêu thi THPT, 70 chỉ tiêu xét tuyển kết hợp, 40 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Công nghệ thông tin chất lượng cao tại cơ sở phía Nam (BVS) (Mã ngành 7480201_CLC) là 100 gồm 10 chỉ tiêu xét tuyển tài năng, 50 chỉ tiêu thi THPT, 25 chỉ tiêu xét tuyển kết hợp, 15 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành An toàn thông tin tại cơ sở phía Bắc (BVH) (Mã ngành 7480202) là 280 gồm 30 chỉ tiêu xét tuyển tài năng, 140 chỉ tiêu thi THPT, 70 chỉ tiêu xét tuyển kết hợp, 40 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành An toàn thông tin tại cơ sở phía Nam (BVS) (Mã ngành 7480202) là 80 gồm 10 chỉ tiêu xét tuyển tài năng, 40 chỉ tiêu thi THPT, 20 chỉ tiêu xét tuyển kết hợp, 10 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Khoa học máy tính định hướng Khoa học dữ liệu tại cơ sở phía Bắc (BVH) (Mã ngành 7480101) là 140 gồm 15 chỉ tiêu xét tuyển tài năng, 70 chỉ tiêu thi THPT, 35 chỉ tiêu xét tuyển kết hợp, 20 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Mạng máy tính và truyền thông dữ liệu (CT Kỹ thuật dữ liệu) tại cơ sở phía Bắc (BVH) (Mã ngành 7480102) là 100 gồm 10 chỉ tiêu xét tuyển tài năng, 50 chỉ tiêu thi THPT, 25 chỉ tiêu xét tuyển kết hợp, 15 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Công nghệ đa phương tiện tại cơ sở phía Bắc (BVH) (Mã ngành 7329001) là 240 gồm 25 chỉ tiêu xét tuyển tài năng, 120 chỉ tiêu thi THPT, 60 chỉ tiêu xét tuyển kết hợp, 35 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Công nghệ đa phương tiện tại cơ sở phía Nam (BVS) (Mã ngành 7329001) là 120 gồm 10 chỉ tiêu xét tuyển tài năng, 60 chỉ tiêu thi THPT, 30 chỉ tiêu xét tuyển kết hợp, 20 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Truyền thông đa phương tiện tại cơ sở phía Bắc (BVH) (Mã ngành 7320104) là 140 gồm 15 chỉ tiêu xét tuyển tài năng, 70 chỉ tiêu thi THPT, 35 chỉ tiêu xét tuyển kết hợp, 20 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Báo chí tại cơ sở phía Bắc (BVH) (Mã ngành 7320101) là 80 gồm 10 chỉ tiêu xét tuyển tài năng, 40 chỉ tiêu thi THPT, 20 chỉ tiêu xét tuyển kết hợp, 10 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Quản trị kinh doanh tại cơ sở phía Bắc (BVH) (Mã ngành 7340101) là 230 gồm 25 chỉ tiêu xét tuyển tài năng, 115 chỉ tiêu thi THPT, 55 chỉ tiêu xét tuyển kết hợp, 35 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Quản trị kinh doanh tại cơ sở phía Nam (BVS) (Mã ngành 7340101) là 100 gồm 10 chỉ tiêu xét tuyển tài năng, 50 chỉ tiêu thi THPT, 25 chỉ tiêu xét tuyển kết hợp, 15 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Thương mại điện tử tại cơ sở phía Bắc (BVH) (Mã ngành 7340122) là 150 gồm 20 chỉ tiêu xét tuyển tài năng, 75 chỉ tiêu thi THPT, 35 chỉ tiêu xét tuyển kết hợp, 20 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Marketing tại cơ sở phía Bắc (BVH) (Mã ngành 7340115) là 220 gồm 25 chỉ tiêu xét tuyển tài năng, 110 chỉ tiêu thi THPT, 55 chỉ tiêu xét tuyển kết hợp, 30 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Marketing tại cơ sở phía Nam (BVS) (Mã ngành 7340115) là 90 gồm 10 chỉ tiêu xét tuyển tài năng, 45 chỉ tiêu thi THPT, 20 chỉ tiêu xét tuyển kết hợp, 15 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Marketing chất lượng cao tại cơ sở phía Bắc (BVH) (Mã ngành 7340115_CLC) là 100 gồm 10 chỉ tiêu xét tuyển tài năng, 50 chỉ tiêu thi THPT, 25 chỉ tiêu xét tuyển kết hợp, 15 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Marketing chất lượng cao tại cơ sở phía Nam (BVS) (Mã ngành 7340115_CLC) là 40 gồm 5 chỉ tiêu xét tuyển tài năng, 20 chỉ tiêu thi THPT, 10 chỉ tiêu xét tuyển kết hợp, 5 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Kế toán tại cơ sở phía Bắc (BVH) (Mã ngành 7340301) là 120 gồm 10 chỉ tiêu xét tuyển tài năng, 60 chỉ tiêu thi THPT, 30 chỉ tiêu xét tuyển kết hợp, 20 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Kế toán tại cơ sở phía Nam (BVS) (Mã ngành 7340301) là 90 gồm 10 chỉ tiêu xét tuyển tài năng, 45 chỉ tiêu thi THPT, 20 chỉ tiêu xét tuyển kết hợp, 15 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        chỉ tiêu tuyển sinh ngành Kế toán chuẩn quốc tế ACCA tại cơ sở phía Nam (BVS) (Mã ngành 7340301_CLC) là 100 gồm 10 chỉ tiêu xét tuyển tài năng, 50 chỉ tiêu thi THPT, 25 chỉ tiêu xét tuyển kết hợp, 15 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Công nghệ tài chính Fintech tại cơ sở phía Bắc (BVH) (Mã ngành 7340205) là 130 gồm 15 chỉ tiêu xét tuyển tài năng, 65 chỉ tiêu thi THPT, 30 chỉ tiêu xét tuyển kết hợp, 20 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        chỉ tiêu tuyển sinh ngành Công nghệ Inernet vạn vật tại cơ sở phía Nam (BVS) (Mã ngành 7520208) là 80 gồm 10 chỉ tiêu xét tuyển tài năng, 40 chỉ tiêu thi THPT, 20 chỉ tiêu xét tuyển kết hợp, 10 chỉ tiêu đánh giá năng lực, đánh giá tư duy.
        
        Chỉ tiêu tuyển các Chương trình Liên kết quốc tế:
        chỉ tiêu tuyển sinh ngành Công nghệ thông tin liên kết quốc tế (Liên kết với Đại học La Trobe, Australia) (Mã ngành 7480201_LK): 15 chỉ tiêu 
        chỉ tiêu tuyển sinh ngành Công nghệ tài chính liên kết quốc tế (Liên kết với Đại học Huddersfield, Vương quốc Anh) (Mã ngành 7340205_LK): 15 chỉ tiêu 
        chỉ tiêu tuyển sinh ngành Công nghệ đa phương liên kết quốc tế tiện (Liên kết với Đại học Canberra, Australia) (Mã ngành 7329001_LK): 15 chỉ tiêu""",
        "use_answer": True,
        "answer": [
            {
                "type": "text",
                "content": "Tổng chỉ tiêu tuyển sinh tất cả các ngành năm 2024 dự kiến là 5.060",
            },
            {
                "type": "text",
                "content": """Tại cơ sở phía Bắc (Mã trường: BVH) tổng chỉ tiêu các ngành là 3900 gồm:
- 420 chỉ tiêu xét tuyển tài năng
- 1950 chỉ tiêu thi THPT (Trung học phổ thông)
- 965 chỉ tiêu xét tuyển kết hợp (XTKH)
- 565 chỉ tiêu đánh giá năng lực, đánh giá tư duy (ĐGNL, ĐGTD).
Chi tiết các thông tin chỉ tiêu tương ứng với các ngành như sau:""",
            },
            {
                "type": "text",
                "content": """Chi tiết các thông tin chỉ tiêu tương ứng với các ngành như sau:""",
            },
            {
                "type": "image",
                "content": "https://firebasestorage.googleapis.com/v0/b/light-team-399402.appspot.com/o/chi_tieu_tuyen_sinh.png?alt=media&token=f569d377-8b5c-4811-969b-5de4c04925c9",
            },
            {
                "type": "text",
                "content": """Tại cơ sở phía Nam (Mã trường: BVS) tổng chỉ tiêu các ngành là 1160 gồm:
- 120 chỉ tiêu xét tuyển tài năng
- 575 chỉ tiêu thi THPT (Trung học phổ thông)
- 280 chỉ tiêu xét tuyển kết hợp (XTKH)
- 185 chỉ tiêu đánh giá năng lực, đánh giá tư duy (ĐGNL, ĐGTD).""",
            },
            {
                "type": "text",
                "content": """Chi tiết các thông tin chỉ tiêu tương ứng với các ngành như sau:""",
            },
            {
                "type": "image",
                "content": "https://firebasestorage.googleapis.com/v0/b/light-team-399402.appspot.com/o/chi_tieu_tuyen_sinh_2.png?alt=media&token=50e53eba-1aba-45e0-a3f5-723126abf80c",
            },
            {
                "type": "text",
                "content": """Thông tin chi tiết tại: https://tuyensinh.ptit.edu.vn/thong-bao/thong-bao-tuyen-sinh-dai-hoc-he-chinh-quy-nam-2024-1712824069684?type=Tuy%E1%BB%83n%20sinh%20%C4%91%E1%BA%A1i%20h%E1%BB%8Dc""",
            },
        ],
    },
    {
        "keys": [
            "kết hợp",
            "ielts",
            "toeic",
            "toefl",
            "sat",
            "act",
            "itp",
            "ibt",
            "chứng chỉ tiếng anh",
            "chứng chỉ",
        ],
        "insert_key": [
            "thí sinh xét tuyển kết hợp",
            "thí sinh có chứng chỉ ielts",
            "thí sinh có chứng chỉ toeic",
            "thí sinh có chứng chỉ toefl",
            "thí sinh có chứng chỉ sat",
            "thí sinh có chứng chỉ act",
            "thí sinh có chứng chỉ itp",
            "thí sinh có chứng chỉ ibt",
            "thí sinh có chứng chỉ tiếng anh",
            "thí sinh có chứng chỉ",
        ],
        "requirements": [
            "có thể xét tuyển phương thức xét tuyển nào, điều kiện xét tuyển, quy trình đăng ký"
        ],
        "level": 2,
        "document": """thí sinh xét tuyển phương thức 3-xét tuyển kết hợp và cần đáp ứng một trong các điều kiện sau đây:
        1. Thí sinh có Chứng chỉ quốc tế SAT, thí sinh cần đạt từ 1130/1600 trở lên trong thời hạn 02 năm tính đến ngày xét tuyển hoặc đạt điểm ACT từ 25/36 trở lên kết hợp với có kết quả điểm trung bình chung học tập lớp 10, 11, 12 hoặc học kỳ 1 lớp 12 (nếu chưa có kết quả năm học lớp 12) đạt từ 7,5 trở lên và có hạnh kiểm Khá trở lên.
        2. Thí sinh có chứng chỉ tiếng Anh quốc tế như IELTS 5.5 trở lên, TOEFL iBT 65 trở lên hoặc TOEFL ITP 513 trở lên kết hợp với có kết quả điểm trung bình chung học tập lớp 10, 11, 12 hoặc học kỳ 1 lớp 12 (nếu chưa có kết quả năm học lớp 12) đạt từ 7,5 trở lên và có hạnh kiểm Khá trở lên.

        Quy trình đăng ký khai và nộp hồ sơ trực tuyến như sau:
    Bước 1: Thí sinh đăng ký tài khoản, lựa chọn Phương thức xét tuyển và khai hồ sơ đăng ký xét tuyển trực tuyến tại địa chỉ website: https://xettuyen.ptit.edu.vn
    Bước 2: In 02 Phiếu ĐKXT theo phương thức xét tuyển đã chọn rồi xin xác nhận của trường THPT nơi thí sinh đang học hoặc Công an xã, phường nơi thí sinh tự do đang cư trú tại địa phương
    Bước 3: Chuẩn bị đầy đủ hồ sơ ĐKXT theo yêu cầu của phương thức xét tuyển đã chọn
    Bước 4: Nộp hồ sơ ĐKXT bằng đường bưu điện chuyển phát nhanh hoặc chuyển phát đảm bảo đến các địa chỉ cơ sở đào tạo của Học viện (bao gồm phiếu đăng ký xét tuyển in từ hệ thống và các giấy tờ cần thiết khác) và nhận thông tin kết quả trúng tuyển của Học viện qua địa chỉ email đã đăng ký.""",
        "answer": [
            {"type": "text", "content": "The answer is 42."},
            {
                "type": "image",
                "content": "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png",
            },
        ],
    },
    {
        "keys": [
            "olympic",
            "tuyển thẳng",
            "giải quốc tế",
            "quốc tế",
            "đội tuyển quốc tế",
        ],
        "insert_key": [
            "thí sinh dự thi olympic quốc tế",
            "thí sinh xét tuyển thẳng",
            "thí sinh có giải quốc tế hoặc đã tham gia thi học sinh giỏi quốc tế",
            "thí sinh có giải quốc tế hoặc đã tham gia thi học sinh giỏi quốc tế",
            "thí sinh có giải học sinh giỏi hoặc đã tham gia thi chọn học sinh giỏi quốc tế",
        ],
        "requirements": [
            "có được tuyển thẳng, phương thức xét tuyển, điều kiện xét tuyển, quy trình đăng ký"
        ],
        "level": 2,
        "document": """Thí sinh được học viện xét tuyển thẳng và ưu tiên xét tuyển khi thí sinh đăng ký phương thức 1-xét tuyển tài năng với các thí sinh tốt nghiệp trung học phổ thông năm 2024 và đạt một trong các thành tích sau:
a) Thí sinh tham dự kỳ thi chọn đội tuyển quốc gia dự thi Olympic quốc tế các môn Toán học, Vật lý, Hóa học hoặc Tin học Thí sinh trong đội tuyển quốc gia dự Cuộc thi khoa học, kỹ thuật quốc tế thời gian tham dự kỳ thi chọn đội tuyển quốc gia không quá 3 năm tính tới thời điểm xét tuyển thẳng
b) Thí sinh đạt giải Nhất, Nhì, Ba các môn Toán học, Vật lý, Hóa học hoặc Tin học trong kỳ thi chọn học sinh giỏi quốc gia, quốc tế do Bộ Giáo dục và Đào tạo tổ chức, cử tham gia thời gian đạt giải không quá 3 năm tính tới thời điểm xét tuyển thẳng
c)Thí sinh đạt giải Nhất, Nhì, Ba trong Cuộc thi khoa học, kỹ thuật cấp quốc gia, quốc tế do Bộ Giáo dục và Đào tạo tổ chức, cử tham gia (Căn cứ vào đề tài dự thi của thí sinh đạt giải, Học viện xem xét xét tuyển thẳng thí sinh vào ngành đào tạo phù hợp) thời gian đạt giải không quá 3 năm tính tới thời điểm xét tuyển thẳng
d) Thí sinh đạt giải Nhất, Nhì, Ba trong các kỳ thi tay nghề khu vực ASEAN và thi tay nghề quốc tế do Bộ Lao động – Thương binh và Xã hội cử đi thời gian đạt giải không quá 3 năm tính tới thời điểm xét tuyển thẳng và có kết quả thi THPT năm 2024 theo tổ hợp xét tuyển của ngành đăng ký xét tuyển đạt Ngưỡng đảm bảo chất lượng đầu vào của Học viện (Căn cứ vào lĩnh vực, nghề của thí sinh đạt giải, Học viện xem xét xét tuyển thẳng thí sinh vào ngành đào tạo phù hợp).""",
        "answer": [],
    },
    {
        "keys": [
            "giải quốc gia",
            "đội tuyển quốc gia",
            "quốc gia",
        ],
        "insert_key": [
            "thí sinh có giải học sinh giỏi quốc gia hoặc đã tham gia thi chọn học sinh giỏi quốc gia",
            "thí sinh có giải học sinh giỏi quốc gia hoặc đã tham gia thi chọn học sinh giỏi quốc gia",
            "thí sinh có giải học sinh giỏi quốc gia hoặc đã tham gia thi chọn học sinh giỏi quốc gia",
        ],
        "requirements": [
            "xét tuyển phương thức 1 vớ xét tuyển nào, điều kiện xét tuyển, quy trình đăng ký"
        ],
        "level": 2,
        "document": """Thí sinh có thể đăng ký phương thức 1-xét tuyển tài năng và đáp ứng một trong các điều kiện sau:
1. Xét tuyển thẳng và ưu tiên xét tuyển với thí sinh đạt giải Nhất, Nhì, Ba các môn Toán học, Vật lý, Hóa học hoặc Tin học trong kỳ thi chọn học sinh giỏi quốc gia, quốc tế do Bộ Giáo dục và Đào tạo tổ chức, cử tham gia thời gian đạt giải không quá 3 năm tính tới thời điểm xét tuyển thẳng
2. Xét tuyển dựa vào hồ sơ năng lực (HSNL) với thí sinh đoạt giải Khuyến khích trong kỳ thi chọn học sinh giỏi quốc gia hoặc đã tham gia kỳ thi chọn học sinh giỏi quốc gia và có kết quả điểm trung bình chung học tập lớp 10, 11, 12 hoặc học kỳ 1 lớp 12 (nếu chưa có kết quả năm học lớp 12) đạt từ 7,5 trở lên và có hạnh kiểm Khá trở lên

        Quy trình đăng ký khai và nộp hồ sơ trực tuyến như sau:
        Bước 1: Thí sinh đăng ký tài khoản, lựa chọn Phương thức xét tuyển và khai hồ sơ đăng ký xét tuyển trực tuyến tại địa chỉ website: https://xettuyen.ptit.edu.vn
    Bước 2: In 02 Phiếu ĐKXT theo phương thức xét tuyển đã chọn rồi xin xác nhận của trường THPT nơi thí sinh đang học hoặc Công an xã, phường nơi thí sinh tự do đang cư trú tại địa phương
    Bước 3: Chuẩn bị đầy đủ hồ sơ ĐKXT theo yêu cầu của phương thức xét tuyển đã chọn
    Bước 4: Nộp hồ sơ ĐKXT bằng đường bưu điện chuyển phát nhanh hoặc chuyển phát đảm bảo đến các địa chỉ cơ sở đào tạo của Học viện (bao gồm phiếu đăng ký xét tuyển in từ hệ thống và các giấy tờ cần thiết khác) và nhận thông tin kết quả trúng tuyển của Học viện qua địa chỉ email đã đăng ký.""",
        "answer": [],
    },
    {
        "keys": [
            "giải tỉnh",
            "giải thành phố",
            "tỉnh",
            "thành phố",
            "giải học sinh giỏi",
            "giải",
            "học sinh giỏi",
        ],
        "insert_key": [
            "thí sinh có giải học sinh giỏi cấp tỉnh",
            "thí sinh có giải học sinh giỏi cấp thành phố",
            "thí sinh có giải học sinh giỏi cấp tỉnh",
            "thí sinh có giải học sinh giỏi cấp thành phố",
            "thí sinh có giải học sinh giỏi",
            "thí sinh có giải",
            "thí sinh có giải học sinh giỏi",
        ],
        "requirements": [
            "có thể xét tuyển phương thức xét tuyển nào, điều kiện xét tuyển, quy trình đăng ký"
        ],
        "level": 2,
        "document": """thí sinh xét tuyển phương thức 1-xét tuyển tài năng dựa vào hồ sơ năng lực (HSNL) với điều kiện:
        - đoạt giải Khuyến khích trong kỳ thi chọn học sinh giỏi quốc gia hoặc đã tham gia kỳ thi chọn học sinh giỏi quốc gia 
        - hoặc đoạt giải Nhất, Nhì, Ba, Khuyến khích trong kỳ thi chọn học sinh giỏi cấp Tỉnh, Thành phố trực thuộc Trung ương (TW) các môn Toán, Lý, Hóa, Tin học 
        - và có kết quả điểm trung bình chung học tập lớp 10, 11, 12 hoặc học kỳ 1 lớp 12 (nếu chưa có kết quả năm học lớp 12) đạt từ 7,5 trở lên và có hạnh kiểm Khá trở lên.

        Quy trình đăng ký khai và nộp hồ sơ trực tuyến như sau:
        Bước 1: Thí sinh đăng ký tài khoản, lựa chọn Phương thức xét tuyển và khai hồ sơ đăng ký xét tuyển trực tuyến tại địa chỉ website: https://xettuyen.ptit.edu.vn
    Bước 2: In 02 Phiếu ĐKXT theo phương thức xét tuyển đã chọn rồi xin xác nhận của trường THPT nơi thí sinh đang học hoặc Công an xã, phường nơi thí sinh tự do đang cư trú tại địa phương
    Bước 3: Chuẩn bị đầy đủ hồ sơ ĐKXT theo yêu cầu của phương thức xét tuyển đã chọn
    Bước 4: Nộp hồ sơ ĐKXT bằng đường bưu điện chuyển phát nhanh hoặc chuyển phát đảm bảo đến các địa chỉ cơ sở đào tạo của Học viện (bao gồm phiếu đăng ký xét tuyển in từ hệ thống và các giấy tờ cần thiết khác) và nhận thông tin kết quả trúng tuyển của Học viện qua địa chỉ email đã đăng ký.""",
        "answer": [],
    },
    {
        "keys": [
            "chuyên",
            "trường chuyên",
            "lớp chuyên",
            "trung học phổ thông chuyên",
        ],
        "insert_key": [
            "thí sinh là học sinh chuyên",
            "thí sinh là học sinh trường chuyên",
            "thí sinh là học sinh lớp chuyên",
            "thí sinh là học sinh trung học phổ thông chuyên",
        ],
        "requirements": [
            "có thể xét tuyển phương thức xét tuyển nào, điều kiện xét tuyển, quy trình đăng ký"
        ],
        "level": 2,
        "document": """thí sinh xét tuyển phương thức 1-xét tuyển tài năng dựa vào hồ sơ năng lực (HSNL) với điều điều kiện là học sinh chuyên các môn Toán, Lý, Hóa, Tin học của trường THPT chuyên trên phạm vi toàn quốc (các trường THPT chuyên thuộc Tỉnh, Thành phố trực thuộc TW và các trường THPT chuyên thuộc Cơ sở giáo dục đại học) hoặc hệ chuyên thuộc các trường THPT trọng điểm quốc gia Và có kết quả điểm trung bình chung học tập lớp 10, 11, 12 hoặc học kỳ 1 lớp 12 (nếu chưa có kết quả năm học lớp 12) đạt từ 7,5 trở lên và có hạnh kiểm Khá trở lên.

        Quy trình đăng ký khai và nộp hồ sơ trực tuyến như sau:
        Bước 1: Thí sinh đăng ký tài khoản, lựa chọn Phương thức xét tuyển và khai hồ sơ đăng ký xét tuyển trực tuyến tại địa chỉ website: https://xettuyen.ptit.edu.vn
    Bước 2: In 02 Phiếu ĐKXT theo phương thức xét tuyển đã chọn rồi xin xác nhận của trường THPT nơi thí sinh đang học hoặc Công an xã, phường nơi thí sinh tự do đang cư trú tại địa phương
    Bước 3: Chuẩn bị đầy đủ hồ sơ ĐKXT theo yêu cầu của phương thức xét tuyển đã chọn
    Bước 4: Nộp hồ sơ ĐKXT bằng đường bưu điện chuyển phát nhanh hoặc chuyển phát đảm bảo đến các địa chỉ cơ sở đào tạo của Học viện (bao gồm phiếu đăng ký xét tuyển in từ hệ thống và các giấy tờ cần thiết khác) và nhận thông tin kết quả trúng tuyển của Học viện qua địa chỉ email đã đăng ký.""",
        "answer": [],
    },
    {
        "keys": [
            "tài năng",
            "phương thức 1",
        ],
        "insert_key": [
            "thí sinh xét tuyển tài năng",
            "thí sinh xét tuyển phương thức 1",
        ],
        "requirements": ["có điều kiện xét tuyển, quy trình đăng ký"],
        "level": 2,
        "document": """thí sinh xét tuyển phương thức 1-xét tuyển tài năng gồm có:
1. Xét tuyển thẳng và ưu tiên xét tuyển: thí sinh là thành viên đội tuyển Olympic quốc tế hoặc đoạt giải Quốc gia, Quốc tế theo Quy chế tuyển sinh hiện hành của Bộ Giáo dục và Đào tạo và của Học viện.
2. Xét tuyển dựa vào hồ sơ năng lực (HSNL): thí sinh cần có thêm một trong các điều kiện sau đây:
- Thí sinh đoạt giải Khuyến khích trong kỳ thi chọn học sinh giỏi quốc gia hoặc đã tham gia kỳ thi chọn học sinh giỏi quốc gia
- Hoặc đoạt giải Nhất, Nhì, Ba, Khuyến khích trong kỳ thi chọn học sinh giỏi cấp Tỉnh, Thành phố trực thuộc Trung ương (TW) các môn Toán, Lý, Hóa, Tin học và có kết quả điểm trung bình chung học tập lớp 10, 11, 12 hoặc học kỳ 1 lớp 12 (nếu chưa có kết quả năm học lớp 12) đạt từ 7,5 trở lên và có hạnh kiểm Khá trở lên
- Là học sinh chuyên các môn Toán, Lý, Hóa, Tin học của trường THPT chuyên trên phạm vi toàn quốc (các trường THPT chuyên thuộc Tỉnh, Thành phố trực thuộc TW và các trường THPT chuyên thuộc Cơ sở giáo dục đại học) hoặc hệ chuyên thuộc các trường THPT trọng điểm quốc gia Và có kết quả điểm trung bình chung học tập lớp 10, 11, 12 hoặc học kỳ 1 lớp 12 (nếu chưa có kết quả năm học lớp 12) đạt từ 7,5 trở lên và có hạnh kiểm Khá trở lên.

        Quy trình đăng ký khai và nộp hồ sơ trực tuyến như sau:
        Bước 1: Thí sinh đăng ký tài khoản, lựa chọn Phương thức xét tuyển và khai hồ sơ đăng ký xét tuyển trực tuyến tại địa chỉ website: https://xettuyen.ptit.edu.vn
    Bước 2: In 02 Phiếu ĐKXT theo phương thức xét tuyển đã chọn rồi xin xác nhận của trường THPT nơi thí sinh đang học hoặc Công an xã, phường nơi thí sinh tự do đang cư trú tại địa phương
    Bước 3: Chuẩn bị đầy đủ hồ sơ ĐKXT theo yêu cầu của phương thức xét tuyển đã chọn
    Bước 4: Nộp hồ sơ ĐKXT bằng đường bưu điện chuyển phát nhanh hoặc chuyển phát đảm bảo đến các địa chỉ cơ sở đào tạo của Học viện (bao gồm phiếu đăng ký xét tuyển in từ hệ thống và các giấy tờ cần thiết khác) và nhận thông tin kết quả trúng tuyển của Học viện qua địa chỉ email đã đăng ký.""",
        "answer": [],
    },
    {
        "keys": [
            "cách xét tuyển",
            "hình thức xét tuyển",
            "nguyên tắc xét tuyển",
        ],
        "level": 2,
        "document": "",
        "answer": [],
    },
    {
        "keys": [
            "học bạ",
            "xét học bạ",
        ],
        "insert_key": [
            "thí sinh xét tuyển học bạ",
            "thí sinh xét tuyển học bạ",
        ],
        "requirements": [
            "có thể xét tuyển phương thức xét tuyển nào, điều kiện xét tuyển, quy trình đăng ký"
        ],
        "level": 2,
        "document": """Học viện xét tuyển học bạ theo phương thức 1 là hồ sơ năng lực và phương thức 3 là xét tuyển kết hợp. Đối với phương thức xét tuyển học bạ thí sinh đọc thông báo tuyển sinh ở hai phương thức trên để biết điều kiện được nộp theo thông tin dưới đây:
        - Phương thức 1: thí sinh xét tuyển dựa vào hồ sơ năng lực (HSNL) và đáp ứng 1 trong các điều kiến sau đây:
        1. Thí sinh đoạt giải Khuyến khích trong kỳ thi chọn học sinh giỏi quốc gia hoặc đã tham gia kỳ thi chọn học sinh giỏi quốc gia 
        2. Hoặc đoạt giải Nhất, Nhì, Ba, Khuyến khích trong kỳ thi chọn học sinh giỏi cấp Tỉnh, Thành phố trực thuộc Trung ương (TW) các môn Toán, Lý, Hóa, Tin học và có kết quả điểm trung bình chung học tập lớp 10, 11, 12 hoặc học kỳ 1 lớp 12 (nếu chưa có kết quả năm học lớp 12) đạt từ 7,5 trở lên và có hạnh kiểm Khá trở lên
        3. Là học sinh chuyên các môn Toán, Lý, Hóa, Tin học của trường THPT chuyên trên phạm vi toàn quốc (các trường THPT chuyên thuộc Tỉnh, Thành phố trực thuộc TW và các trường THPT chuyên thuộc Cơ sở giáo dục đại học) hoặc hệ chuyên thuộc các trường THPT trọng điểm quốc gia và có kết quả điểm trung bình chung học tập lớp 10, 11, 12 hoặc học kỳ 1 lớp 12 (nếu chưa có kết quả năm học lớp 12) đạt từ 7,5 trở lên và có hạnh kiểm Khá trở lên.
        - Phương thức 3: thí sinh có thể đăng ký xét tuyển kết hợp và đáp ứng 1 trong các điều kiện sau đây:
        1. Thí sinh có Chứng chỉ quốc tế SAT, thí sinh cần đạt từ 1130/1600 trở lên trong thời hạn 02 năm tính đến ngày xét tuyển hoặc đạt điểm ACT từ 25/36 trở lên kết hợp với có kết quả điểm trung bình chung học tập lớp 10, 11, 12 hoặc học kỳ 1 lớp 12 (nếu chưa có kết quả năm học lớp 12) đạt từ 7,5 trở lên và có hạnh kiểm Khá trở lên.
        2. Thí sinh có chứng chỉ tiếng Anh quốc tế như IELTS 5.5 trở lên, TOEFL iBT 65 trở lên hoặc TOEFL ITP 513 trở lên kết hợp với có kết quả điểm trung bình chung học tập lớp 10, 11, 12 hoặc học kỳ 1 lớp 12 (nếu chưa có kết quả năm học lớp 12) đạt từ 7,5 trở lên và có hạnh kiểm Khá trở lên.
        
        Quy trình đăng ký khai và nộp hồ sơ trực tuyến như sau:
        Bước 1: Thí sinh đăng ký tài khoản, lựa chọn Phương thức xét tuyển và khai hồ sơ đăng ký xét tuyển trực tuyến tại địa chỉ website: https://xettuyen.ptit.edu.vn
    Bước 2: In 02 Phiếu ĐKXT theo phương thức xét tuyển đã chọn rồi xin xác nhận của trường THPT nơi thí sinh đang học hoặc Công an xã, phường nơi thí sinh tự do đang cư trú tại địa phương
    Bước 3: Chuẩn bị đầy đủ hồ sơ ĐKXT theo yêu cầu của phương thức xét tuyển đã chọn
    Bước 4: Nộp hồ sơ ĐKXT bằng đường bưu điện chuyển phát nhanh hoặc chuyển phát đảm bảo đến các địa chỉ cơ sở đào tạo của Học viện (bao gồm phiếu đăng ký xét tuyển in từ hệ thống và các giấy tờ cần thiết khác) và nhận thông tin kết quả trúng tuyển của Học viện qua địa chỉ email đã đăng ký.""",
        "answer": [],
    },
    {
        "keys": [
            "đánh giá",
            "năng lực",
            "tư duy",
        ],
        "insert_key": [
            "thí sinh xét tuyển đánh giá năng lực, đánh giá tư duy",
            "thí sinh xét tuyển đánh giá năng lực, đánh giá tư duy",
            "thí sinh xét tuyển đánh giá năng lực, đánh giá tư duy",
        ],
        "requirements": [
            "có thể xét tuyển phương thức xét tuyển nào, điều kiện xét tuyển, quy trình đăng ký"
        ],
        "level": 2,
        "document": """Để xét tuyển vào Học viện Công nghệ Bưu Chính Viễn thông (PTIT) dựa vào đánh giá năng lực và tư duy, thí sinh có thể xét tuyển theo phương thức 4 và đáp ứng một trong các điều kiện sau:\n- Thí sinh có điểm thi đánh giá năng lực của Đại học quốc gia Hà Nội (HSA) năm 2024 từ 75 điểm trở lên\n- Thí sinh có điểm thi đánh giá năng lực của Đại học quốc gia Tp. Hồ Chí Minh (APT) năm 2024 từ 600 điểm trở lên\n- Thí sinh có điểm thi đánh giá tư duy của Đại học Bách khoa Hà Nội (TSA) năm 2024 từ 50 điểm trở lên.
        
        Quy trình đăng ký khai và nộp hồ sơ trực tuyến như sau:
        Bước 1: Thí sinh đăng ký tài khoản, lựa chọn Phương thức xét tuyển và khai hồ sơ đăng ký xét tuyển trực tuyến tại địa chỉ website: https://xettuyen.ptit.edu.vn
    Bước 2: In 02 Phiếu ĐKXT theo phương thức xét tuyển đã chọn rồi xin xác nhận của trường THPT nơi thí sinh đang học hoặc Công an xã, phường nơi thí sinh tự do đang cư trú tại địa phương
    Bước 3: Chuẩn bị đầy đủ hồ sơ ĐKXT theo yêu cầu của phương thức xét tuyển đã chọn
    Bước 4: Nộp hồ sơ ĐKXT bằng đường bưu điện chuyển phát nhanh hoặc chuyển phát đảm bảo đến các địa chỉ cơ sở đào tạo của Học viện (bao gồm phiếu đăng ký xét tuyển in từ hệ thống và các giấy tờ cần thiết khác) và nhận thông tin kết quả trúng tuyển của Học viện qua địa chỉ email đã đăng ký.""",
        "answer": [],
    },
    {
        "keys": [
            "khai hồ sơ",
            "nộp hồ sơ",
            "đăng ký",
            "hướng dẫn",
        ],
        "insert_key": [
            "hướng dẫn khai hồ sơ xét tuyển",
            "hướng dẫn nộp hồ sơ xét tuyển",
            "hướng dẫn đăng ký xét tuyển",
            "hướng dẫn đăng ký khai và nộp hồ sơ xét tuyển",
        ],
        "requirements": [""],
        "level": 3,
        "document": """Hướng dẫn đăng ký khai và nộp hồ sơ xét tuyển như sau:
        Bước 1: Thí sinh đăng ký tài khoản, lựa chọn Phương thức xét tuyển và khai hồ sơ đăng ký xét tuyển trực tuyến tại địa chỉ website: https://xettuyen.ptit.edu.vn
    Bước 2: In 02 Phiếu ĐKXT theo phương thức xét tuyển đã chọn rồi xin xác nhận của trường THPT nơi thí sinh đang học hoặc Công an xã, phường nơi thí sinh tự do đang cư trú tại địa phương
    Bước 3: Chuẩn bị đầy đủ hồ sơ ĐKXT theo yêu cầu của phương thức xét tuyển đã chọn
    Bước 4: Nộp hồ sơ ĐKXT bằng đường bưu điện chuyển phát nhanh hoặc chuyển phát đảm bảo đến các địa chỉ cơ sở đào tạo của Học viện (bao gồm phiếu đăng ký xét tuyển in từ hệ thống và các giấy tờ cần thiết khác) và nhận thông tin kết quả trúng tuyển của Học viện qua địa chỉ email đã đăng ký.""",
        "answer": [],
    },
]


def rewrite_question(question: str):
    question = normalize_replace_abbreviation_text(question)
    print(question)
    docs_keys = []
    for i in data_map_key:
        for key in i["keys"]:
            # print("check keywork", key)
            if (
                key in question
                and i["document"] not in docs_keys
                and i["document"] != ""
            ):
                print("check keywork", key)
                if "insert_key" in i:
                    question = i["insert_key"][i["keys"].index(key)]
                if "requirements" in i:
                    question += " " + i["requirements"][0]
                docs_keys.append(
                    Document(
                        page_content=i["document"], metadata={"source": "data_map_key"}
                    )
                )
                break
    return question


def normalize_replace_abbreviation_text(text):
    # text = re.sub(
    #     r"[\.,\(\)]", " ", text
    # )  # thay thế các kí tự đặc biệt bằng khoảng trắng
    # text = re.sub("<.*?>", "", text).strip()
    # text = re.sub("(\s)+", r"\1", text)
    # chars = re.escape(string.punctuation)
    # text = re.sub(
    #     r"[" + chars + "]", " ", text
    # )  # thay thế các kí tự đặc biệt bằng khoảng trắng
    text = re.sub(r"\s+", " ", text)  # thay thế nhiều khoảng trắng bằng 1 khoảng trắng
    text = text.strip()  # xóa khoảng trắng ở đầu và cuối
    text = text.lower()  # chuyển về chữ thường
    """ 
    # "cntt" -> "công nghệ thông tin"
    text = re.sub(r'\bcntt\b', 'công nghệ thông tin', text)
    # "ntn" -> "như thế nào"
    text = re.sub(r'\bntn\b', 'như thế nào', text)
    # "ad, adm" -> "admin"
    text = re.sub(r'\b(ad|adm)\b', 'admin', text)
    text = re.sub(r'\b(gd dt|gddt)\b', 'giáo dục đào tạo', text) 
    # điểm chuẩn -> điểm trúng tuyển
    text = re.sub(r'\bđiểm chuẩn\b', 'điểm trúng tuyển', text)
    """

    for k, v in dict_replace.items():
        text = re.sub(r"\b" + "(" + k + ")" + r"\b", v, text)

    text = text.lower()

    return text


def rewrite_question(question: str):
    question = normalize_replace_abbreviation_text(question)
    print(question)
    docs_keys = []
    for i in data_map_key:
        for key in i["keys"]:
            # print("check keywork", key)
            if key in question and i["document"] not in docs_keys:
                if "requirements" in i:
                    question = question + " " + i["requirements"][0]
                docs_keys.append(
                    Document(
                        page_content=i["document"], metadata={"source": "data_map_key"}
                    )
                )
                break
    return question


# Define your desired data structure.
class Test(BaseModel):
    query: str = Field(description="Câu hỏi của người dùng")
    answer: str = Field(description="Câu trả lời của mô hình")
    question_1: str = Field(description="Câu hỏi thường gặp 1, được tạo từ văn bản")
    answer_1: str = Field(description="Câu trả lời của câu hỏi thường gặp 1")
    question_2: str = Field(description="Câu hỏi thường gặp 2, được tạo từ văn bản")
    answer_2: str = Field(description="Câu trả lời của câu hỏi thường gặp 2")
    # Set up a parser + inject instructions into the prompt template.


# And a query intented to prompt a language model to populate the data structure.
parser = JsonOutputParser(pydantic_object=Test)


RESPONSE_TEMPLATE = """\
Bạn là một lập trình viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của Học viện Công nghệ Bưu Chính Viễn thông (PTIT).

Tạo câu trả lời đầy đủ và đầy đủ thông tin từ 180 từ trở xuống cho \
câu hỏi đưa ra chỉ dựa trên kết quả tìm kiếm được cung cấp (Document và nội dung). Bạn phải \
chỉ sử dụng thông tin từ kết quả tìm kiếm được cung cấp. Sử dụng một cách khách quan và \
giọng điệu báo chí. Kết hợp các kết quả tìm kiếm lại với nhau thành một câu trả lời mạch lạc, thân thiện nhất với người dùng. Đừng \
lặp lại văn bản. Nếu như các kết quả khác nhau đề cập đến các thực thể khác nhau trong cùng một tên, hãy viết \
câu trả lời cho từng thực thể.

Bạn nên tách đoạn trong câu trả lời, sử dụng dấu đầu dòng khi cần thiết trong câu trả lời của mình để dễ đọc. 

Nếu không có gì trong ngữ cảnh liên quan đến câu hỏi hiện tại, bạn chỉ cần nói "Hmm, tôi không chắc." Đừng cố bịa ra một câu trả lời. Đừng cố khẳng định bất kì điều gì.

Mọi thứ giữa các khối `context` sau đây đều được lấy từ một kiến thức \
ngân hàng, không phải là một phần của cuộc trò chuyện với người dùng.

<context>
     {context}
<context/>

HÃY NHỚ: Nếu <context> chứa cả thông tin phía Bắc và phía Nam, bạn cần phải trích dẫn cả hai phần và ưu tiên trình bày thông tin phía Bắc rồi đến phía Nam.\
Nếu không có thông tin liên quan trong ngữ cảnh, bạn chỉ cần nói "Hmm, tôi không chắc." Đừng cố bịa ra một câu trả lời. Đừng cố khẳng định bất kì điều gì. Bất cứ điều gì nằm giữa 'context' trước đó \
các Document và nội dung được lấy từ ngân hàng kiến thức, không phải là một phần của cuộc trò chuyện với \
người dùng.\
"""

COHERE_RESPONSE_TEMPLATE = """\
Bạn là một lập trình viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của Học viện Công nghệ Bưu Chính Viễn thông (PTIT).

Tạo câu trả lời đầy đủ và đầy đủ thông tin từ 180 từ trở xuống cho \
câu hỏi đưa ra chỉ dựa trên kết quả tìm kiếm được cung cấp (Document và nội dung). Bạn phải \
chỉ sử dụng thông tin từ kết quả tìm kiếm được cung cấp. Sử dụng một cách khách quan và \
giọng điệu báo chí. Kết hợp các kết quả tìm kiếm lại với nhau thành một câu trả lời mạch lạc, thân thiện nhất với người dùng. Đừng \
lặp lại văn bản. Nếu như các kết quả khác nhau đề cập đến các thực thể khác nhau trong cùng một tên, hãy viết \
câu trả lời cho từng thực thể.

Bạn nên tách đoạn trong câu trả lời, sử dụng dấu đầu dòng khi cần thiết trong câu trả lời của mình để dễ đọc. 

Nếu không có gì trong ngữ cảnh liên quan đến câu hỏi hiện tại, bạn chỉ cần nói "Hmm, \
tôi không chắc" Đừng cố bịa ra một câu trả lời. Đừng cố khẳng định bất kì điều gì.


HÃY NHỚ: Nếu <context> chứa cả thông tin phía Bắc và phía Nam, bạn cần phải trích dẫn cả hai phần và ưu tiên trình bày thông tin phía Bắc rồi đến phía Nam.\
Nếu không có thông tin liên quan trong ngữ cảnh, bạn chỉ cần nói "Hmm, tôi \
không chắc chắn." Đừng cố bịa ra một câu trả lời. Đừng cố khẳng định bất kì điều gì. Bất cứ điều gì nằm giữa 'context' trước đó \
các Document và nội dung được lấy từ ngân hàng kiến thức, không phải là một phần của cuộc trò chuyện với \
người dùng.\
"""
COHERE_RESPONSE_TEMPLATE = """Từ câu hỏi và văn bản sau đây, hãy trả lời câu hỏi dựa trên văn bản và Tạo thêm 1 đến 2 Câu hỏi thường gặp nhất định (các câu hỏi thường gặp) từ văn bản. Tạo câu hỏi, câu trả lời và ngữ cảnh tương ứng.\
    Nếu không thể tạo câu hỏi từ văn bản, hãy nói "Hmm, tôi không chắc.". Đừng cố gắng tạo câu trả lời, câu hỏi liên quan từ văn bản không phù hợp hoặc không chính xác.\
        .\n{format_instructions}"""

REPHRASE_TEMPLATE = """\
Với cuộc trò chuyện sau đây và một câu hỏi tiếp theo, hãy diễn đạt lại câu tiếp theo \
câu hỏi là một câu hỏi độc lập.

Lịch sử trò chuyện:
{chat_history}
Đầu vào tiếp theo: {question}
Câu hỏi độc lập:"""


# client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def get_retriever() -> BaseRetriever:
    vectorstore = FAISS.load_local(
        "./NewData04022024/vectorDB_05282024/",
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "threshold": 0.5})
    return retriever


dict_vectorstore = {
    "thoi_gian": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/thoi_gian",
    "ho_so": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/ho_so",
    "ban_tuyen_sinh": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/ban_tuyen_sinh",
    "cach_thuc_xet_tuyen_cac_phuong_thuc": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/cach_thuc_xet_tuyen_cac_phuong_thuc",
    "chinh_sach_ho_tro,_uu_tien": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/chinh_sach_ho_tro,_uu_tien",
    "chi_tieu_tuyen_sinh": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/chi_tieu_tuyen_sinh",
    "co_hoi_viec_lam": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/co_hoi_viec_lam",
    "diem_trung_tuyen": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/diem_trung_tuyen",
    "doi_tuong_uu_tien,_khu_vuc_uu_tien": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/doi_tuong_uu_tien,_khu_vuc_uu_tien",
    "giai_hoc_sinh_gioi": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/giai_hoc_sinh_gioi",
    "hoc_bong": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/hoc_bong",
    "hoc_phi": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/hoc_phi",
    "hop_tac_doi_ngoai": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/hop_tac_doi_ngoai",
    "khac": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/khac",
    "nganh_chuyen_nganh": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/nganh/chuyen_nganh",
    "phuong_thuc_1": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/phuong_thuc_1",
    "phuong_thuc_2": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/phuong_thuc_2",
    "phuong_thuc_3": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/phuong_thuc_3",
    "phuong_thuc_4": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/phuong_thuc_4",
    "phuong_thuc_xet_tuyen": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/phuong_thuc_xet_tuyen",
    "quy_doi_chung_chi": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/quy_doi_chung_chi",
    "thong_tin_truong": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/thong_tin_truong",
    "xet_hoc_ba": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/xet_hoc_ba",
    "xet_tuyen_thang": "./NewData04022024/CodeIntentRAG/VectorDB/IntentOutline/xet_tuyen_thang",
}

for key, value in dict_vectorstore.items():
    vectorstore = FAISS.load_local(
        value,
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    dict_vectorstore[key] = vectorstore.as_retriever(
        search_kwargs={"k": 3, "threshold": 0.5}
    )

from ingest import get_retriever, get_intent_retriever


def create_retriever_chain(
    llm: LanguageModelLike, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


from langchain.output_parsers import ResponseSchema, StructuredOutputParser


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="RetrieveDocs")
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            # HumanMessagePromptTemplate.from_template("{question}"),
        ],
    )

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    default_response_synthesizer = prompt | llm | parser

    cohere_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COHERE_RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ],
    )

    cohere_prompt = cohere_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    @chain
    def cohere_response_synthesizer(input: dict) -> RunnableSequence:
        return cohere_prompt | llm.bind(source_documents=input["docs"])

    response_synthesizer = (
        default_response_synthesizer.configurable_alternatives(
            ConfigurableField("llm"),
            default_key="openai_gpt_3_5_turbo",
            anthropic_claude_3_sonnet=default_response_synthesizer,
            fireworks_mixtral=default_response_synthesizer,
            google_gemini_pro=default_response_synthesizer,
            cohere_command=cohere_response_synthesizer,
        )
        | parser
    ).with_config(run_name="GenerateResponse")

    return (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | context
        | response_synthesizer
    )


import os

# load openai api key from file .env
from dotenv import load_dotenv

load_dotenv()

gpt_3_5 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)

cohere_command = ChatCohere(
    model="command",
    temperature=0,
    cohere_api_key=os.environ.get("COHERE_API_KEY", "not_provided"),
)
llm = gpt_3_5.configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="llm"),
    default_key="openai_gpt_3_5_turbo",
    cohere_command=cohere_command,
).with_fallbacks([gpt_3_5, cohere_command])

retriever = get_retriever()
if os.path.exists("data_138"):
    retriever = get_retriever()
    dict_vectorstore = get_intent_retriever()
# answer_chain = create_chain(llm, retriever)

from operator import itemgetter
from typing import List, Tuple

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    HumanMessagePromptTemplate,
)

RESPONSE_TEMPLATE = """Bạn là một tư vấn viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của Học viện Công nghệ Bưu Chính Viễn thông (PTIT).

Từ câu hỏi và văn bản sau đây, hãy trả lời câu hỏi thật chi tiết và chính xác nhất có thể dựa trên thông tin có trong văn bản. 
Bạn nên tách đoạn trong câu trả lời, sử dụng dấu đầu dòng khi cần thiết trong câu trả lời của mình để dễ đọc. 

Nếu thông tin trong ngữ cảnh không liên quan đến câu hỏi hiện tại, bạn chỉ cần nói "Hmm, \
tôi không chắc". Đừng cố bịa ra một câu trả lời và đừng cố khẳng định bất kì điều gì.

Question: {question}\n<context> {context} </context>\n---\n
    HÃY NHỚ: Nếu nội dung <context> chứa cả thông tin liên quan đến phía Bắc và phía Nam, bạn cần phải trích dẫn thật chi tiết, ưu tiên trình bày thông tin phía Bắc rồi đến phía Nam.\
    Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng Việt.
    Với mỗi câu trả lời có câu hỏi giống nhau, bạn cần trả lời giống nhau.
    ---
    Output:

"""

RESPONSE_TEMPLATE = """Bạn là một tư vấn viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của Học viện Công nghệ Bưu Chính Viễn thông (PTIT).

Từ câu hỏi và văn bản sau đây, hãy trả lời câu hỏi thật chi tiết và chính xác nhất có thể dựa trên thông tin có trong văn bản.
Cùng với đó hãy Tạo thêm 1 đến 2 Câu hỏi thường gặp nhất định (các câu hỏi thường gặp) từ văn bản với độ dài.

Câu trả lời được trình bày theo cấu trúc Object JSON dựa theo mô tả bên dưới bao gồm 1 ví dụ mẫu và 1 Schema mô tả các key, value của Object JSON trả về.\n{format_instructions}\nQuestion: {question}\n<context> {context} </context>\n---
HÃY NHỚ: Nếu nội dung <context> chứa cả thông tin liên quan đến phía Bắc và phía Nam, bạn cần phải trích dẫn thật chi tiết, ưu tiên trình bày thông tin theo thứ tự phía Bắc rồi đến phía Nam.\
Nội dung trong "answer" phải trả lời bằng tiếng Việt, tất cả câu trả lời của bạn đều phải trả lời bằng tiếng Việt.
Với mỗi câu trả lời có câu hỏi giống nhau, bạn cần trả lời giống nhau.
Bạn nên tách đoạn trong câu trả lời nếu độ dài câu trả lời trong "answer" quá dài (lớn hơn 30 từ), sử dụng dấu đầu dòng khi cần thiết trong câu trả lời của mình để dễ đọc.
Nếu không có gì trong ngữ cảnh liên quan đến câu hỏi hiện tại, bạn chỉ cần nói "Hmm, tôi không chắc" và đặt "answer": "Hmm, tôi không chắc". Đừng cố bịa ra một câu trả lời không đúng. Đừng cố khẳng định bất kì điều gì.
---
Output:"""
RESPONSE_TEMPLATE = """Bạn là một tư vấn viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của Học viện Công nghệ Bưu Chính Viễn thông (PTIT).

Từ câu hỏi và văn bản sau đây, hãy trả lời câu hỏi thật chi tiết và chính xác nhất có thể dựa trên thông tin có trong văn bản. 
Bạn nên tách đoạn trong câu trả lời, sử dụng dấu đầu dòng khi cần thiết trong câu trả lời của mình để dễ đọc. 

Nếu thông tin trong ngữ cảnh không liên quan đến câu hỏi hiện tại, bạn chỉ cần nói "Hmm, \
tôi không chắc". Đừng cố bịa ra một câu trả lời và đừng cố khẳng định bất kì điều gì.

Question: {question}\n<context> {context} </context>\n---\n
    HÃY NHỚ: Nếu nội dung <context> chứa cả thông tin liên quan đến phía Bắc và phía Nam, bạn cần phải trích dẫn thật chi tiết, ưu tiên trình bày thông tin phía Bắc rồi đến phía Nam.\
    Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng Việt.
    Với mỗi câu trả lời có câu hỏi giống nhau, bạn cần trả lời giống nhau.
    ---
    Output:

"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RESPONSE_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
        # HumanMessagePromptTemplate.from_template("{question}"),
    ],
)

# prompt = prompt.partial(
#     format_instructions=parser.get_format_instructions(),
# )

# Condense a chat history and follow-up question into a standalone question
_template = """\
Với cuộc trò chuyện sau đây và một câu hỏi tiếp theo, hãy diễn đạt lại câu tiếp theo \
câu hỏi là một câu hỏi độc lập. Nếu thông tin Lịch sử trò chuyện không liên quan đến Câu hỏi tiếp theo, hãy sử dụng câu hỏi tiếp theo làm Câu hỏi độc lập.

Lịch sử trò chuyện:
{chat_history}
Câu hỏi tiếp theo: {question}
Câu hỏi độc lập:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# RAG answer synthesis prompt
template = """Bạn là một tư vấn viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của Học viện Công nghệ Bưu Chính Viễn thông (PTIT).

Từ câu hỏi và văn bản sau đây, hãy trả lời câu hỏi thật chi tiết và chính xác nhất có thể dựa trên thông tin có trong văn bản.
Bạn nên tách đoạn trong câu trả lời, sử dụng dấu đầu dòng khi cần thiết trong câu trả lời của mình để dễ đọc.

Nếu không có gì trong ngữ cảnh liên quan đến câu hỏi hiện tại, bạn chỉ cần nói "Hmm, \
tôi không chắc" Đừng cố bịa ra một câu trả lời. Đừng cố khẳng định bất kì điều gì.

Question: {question}\n<context> {context} </context>\n---\n
    HÃY NHỚ: Nếu nội dung <context> chứa cả thông tin liên quan đến phía Bắc và phía Nam, bạn cần phải trích dẫn thật chi tiết, ưu tiên trình bày thông tin phía Bắc rồi đến phía Nam.\
    Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng Việt.
    ---
    Output:"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        # ("system", template),
        ("system", RESPONSE_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# Conversational Retrieval Chain
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

from difflib import SequenceMatcher


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    # docs.sort(key=lambda x: len(x.page_content), reverse=True)

    # for i in range(len(docs)):
    #     text = docs[i].page_content
    #     text = re.sub(r"\n", "|@#ript", text)
    #     text = re.sub(r"\s+", " ", text)
    #     text = re.sub(r"\|@#ript", "\n", text)
    #     text = re.sub(r" ,", "", text)

    #     docs[i].page_content = text

    # l = []
    # st = docs[0].page_content.strip()
    # for i in range(1, len(docs)):
    #     if docs[i].page_content.strip() not in st:
    #         l.append(docs[i])
    #         st += "\n"
    #         st += docs[i].page_content.strip()
    #         print(docs[i])
    # print(len(l))
    # doc_strings = [format_document(doc, document_prompt) for doc in docs]
    # return document_separator.join(doc_strings)

    # remove one of the pair of similar documents > 90% similarity

    # for i in range(len(docs)-1):
    #     for j in range(i+1, len(docs)):
    #         print(i, j)
    #         print(SequenceMatcher(None, docs[i].page_content, docs[j].page_content).ratio())
    #         if SequenceMatcher(None, docs[i].page_content, docs[j].page_content).ratio() > 0.9:
    #             docs[j].page_content = ""
    # [print(i) for i in docs]
    # docs = [i for i in docs if i.page_content != ""]
    # print(len(docs))
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    if len(converted_chat_history) > 3:
        return converted_chat_history[:3]
    return converted_chat_history


# User input
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        # | parser,
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(itemgetter("question")),
)

# _inputs = RunnableParallel(
#     {
#         "question": lambda x: x["question"],
#         "chat_history": lambda x: _format_chat_history(x["chat_history"]),
#         "context": _search_query | retriever | _combine_documents,
#     }
# ).with_types(input_type=ChatHistory)

# chain = _inputs | ANSWER_PROMPT | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) | StrOutputParser()

# FUSION
template_vietnamese_fusion = """Bạn là một tư vấn viên chuyên nghiệp và là người giải quyết vấn đề, được giao nhiệm vụ trả lời bất kỳ câu hỏi nào \
về các thông tin tuyển sinh của Học viện Công nghệ Bưu Chính Viễn thông (PTIT).
Bạn có thể tạo ra nhiều truy vấn tìm kiếm dựa trên một truy vấn đầu vào duy nhất. \n
Tạo ra nhiều truy vấn tìm kiếm liên quan đến: {question} \n
Đầu ra (3 truy vấn):"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template_vietnamese_fusion)
generate_queries = (
    prompt_rag_fusion
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

from langchain.load import dumps, loads


def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula"""

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k).
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


# retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
retrieval_chain_rag_fusion = generate_queries | retriever.map()

import requests


def get_intent(text):
    url = "http://localhost:5080/model/parse"
    payload = {"text": text}
    response = requests.post(url, json=payload)
    return response.json()


def rewrite_question_keword(question: str):
    question = normalize_replace_abbreviation_text(question)
    # data_map_key.sort(key=lambda x: x["level"])
    docs_keys = []
    check_rewrited = False

    for i in data_map_key:
        for key in i["keys"]:
            # print("check keywork", key)
            if (
                key in question
                and i["document"] not in docs_keys
                and i["document"] != ""
            ):
                print("check keywork", key, "-", question)
                if "insert_key" in i and not check_rewrited:
                    if len(i["insert_key"]) > 0:
                        question = i["insert_key"][i["keys"].index(key)]
                    print("question_insert_key", question)
                    if "requirements" in i:
                        question += " " + i["requirements"][0]
                        print("question_requirements", question)
                    check_rewrited = True
                docs_keys.append(i["document"])
                break
    return question


def get_results(question: str):
    question = normalize_replace_abbreviation_text(question)

    # find document by keywork
    question_keword = rewrite_question_keword(question)
    print(question)
    docs_keys = []
    for i in data_map_key:
        text_insert = ""
        for key in i["keys"]:
            # print("check keywork", key)
            if key in question_keword:
                if "use_answer" in i and i["use_answer"] == True:
                    print(i["answer"])
                    return {"use_answer": True, "docs": i["answer"]}
            if key in question_keword and i["document"] not in docs_keys:
                print("check keywork", key)
                index = i["keys"].index(key)
                if "insert_key" in i:
                    if len(i["insert_key"]) > 0:
                        text_insert += i["insert_key"][index] + " "
                        print("text_insert", text_insert)
                        docs_keys.append(
                            Document(
                                page_content=text_insert + i["document"],
                                metadata={"source": "data_map_key"},
                            )
                        )
                    else:
                        docs_keys.append(
                            Document(
                                page_content=i["document"],
                                metadata={"source": "data_map_key"},
                            )
                        )
                break
    # docs = []
    # response = get_intent(question)
    # response['intent_ranking']

    # top_5_intent = response['intent_ranking'][:4]
    # top_5_intent

    # list_docs = []
    # for i in top_5_intent:
    #     if "nlu_fallback" in i['name']:
    #         continue
    #     else:
    #         intent = i['name']
    #         for j in list_paths:
    #             for key, value in j.items():
    #                 if intent == value:
    #                     with open(key, "r", encoding="utf-8") as f:
    #                         doc = f.read()
    #                         # list_docs.append(doc)
    #                         list_docs.append(Document(page_content=doc, metadata={'source': key}))

    # print("docs", docs)
    # print("docs1", docs1)
    # print("list_docs", list_docs)
    if docs_keys:
        return {"use_answer": False, "docs": docs_keys}
    else:
        response = get_intent(question)
        response["intent_ranking"]

        top_5_intent = response["intent_ranking"][:3]
        print(top_5_intent)
        list_docs = []
        for i in top_5_intent:
            if "nlu_fallback" in i["name"]:
                continue
            else:
                intent = i["name"]
                confidence = i["confidence"]
                for key, value in dict_vectorstore.items():
                    if intent == key:
                        print(intent, key)
                        vectorstore = dict_vectorstore[key]
                        docs1 = vectorstore.get_relevant_documents(question)
                        list_docs.append(docs1)

        docs = retrieval_chain_rag_fusion.invoke({"question": question})

        docs1 = retriever.get_relevant_documents(question)
        docs.append(docs1)
        # docs = docs + list_docs
        docs = reciprocal_rank_fusion(docs)

        docs_copy = docs.copy()
        docs_copy.sort(key=lambda x: len(x[0].page_content), reverse=True)
        combined_docs = []
        string_check = ""
        for doc in docs_copy:
            if doc[0].page_content not in string_check:
                string_check += doc[0].page_content
                combined_docs.append(doc)
        combined_docs

        return {"use_answer": False, "docs": combined_docs}


_inputs = RunnableParallel(
    {
        "question": RunnableLambda(itemgetter("question"))
        | normalize_replace_abbreviation_text,
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": RunnableLambda(itemgetter("question")) | get_results,
    }
).with_types(input_type=ChatHistory)

chain_json = (
    _inputs
    | prompt
    | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    | StrOutputParser()
)


def chat_udu(data: any):
    print(data)
    data.text = data.text["text"]
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    prompt = """Dựa vào yêu cầu sau hãy trả lời chi tiết nhất có thể: {re} {question} \n
    Đưa ra câu trả lời:"""

    prompt = ChatPromptTemplate.from_template(prompt)

    chain = prompt | llm

    res = chain.invoke({"re": data.prompt, "question": data.text})
    print(res)
    return res.content
