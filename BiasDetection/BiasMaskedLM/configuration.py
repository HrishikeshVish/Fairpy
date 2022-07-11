configuration = {
    'ko': {
        'bert_model': 'beomi/kcbert-base',
        # 'bert_model': 'bert-base-multilingual-uncased',
        'nationality': ['미국', '캐나다', '일본', '중국', '한국', '영국', '프랑스', '독일', '멕시코', '이라크', '아일랜드', '이란',
                        '사우디', '러시아', '베트남', '태국', '호주', '스페인', '터키', '이스라엘', '이탈리아', '이집트', '캄보디아',
                        '인도', '브라질', '콜롬비아', '그리스', '아프간', '쿠바', '시리아', '북한', '파키스탄', '루마니아', '스위스', '모로코'],
        'template_path': 'templates/templates_ko.txt',
        'occ_path': 'templates/occ_ko.txt',
        'MSK': 4
        # 'MSK': 103
    },
    'en': {
        'bert_model': 'bert-base-uncased',
        # 'bert_model': 'bert-base-multilingual-uncased',
        'nationality': ['america', 'canada', 'japan', 'china', 'korea', 'england',
                        'france', 'germany', 'mexico', 'iraq', 'ireland', 'iran', 'saudi',
                        'russia', 'vietnam', 'thailand', 'australia', 'spain', 'turkey',
                        'israel', 'italy', 'egypt', 'somalia', 'india', 'brazil',
                        'colombia', 'greece', 'afghanistan', 'cuba', 'syria', 'north korea', 'pakistan', 'romania', 'switzerland', 'morocco'],
        'template_path': 'templates/templates_en.txt',
        'occ_path': 'templates/occ_en.txt',
        'MSK': 103
    },
    'de':{
        'bert_model': 'bert-base-german-dbmdz-uncased',
        # 'bert_model': 'bert-base-multilingual-uncased',
        'nationality': ['amerika', 'kanada', 'japan', 'china', 'korea', 'england',
                        'frankreich', 'deutschland', 'mexiko', 'irak', 'irland', 'iran',
                        'saudi', 'russland', 'vietnam', 'thailand', 'australien',
                        'spanien', 'türkei', 'israel', 'italien', 'ägypten', 'somalia',
                        'indien', 'brasilien', 'kolumbien', 'griechenland', 'afghanistan',
                        'kuba', 'syrien', 'nord korea', 'Pakistan', 'Rumänien', 'Schweiz', 'Marokko'],
        'template_path': 'templates/templates_de.txt',
        'occ_path': 'templates/occ_de.txt',
        'MSK': 104
        # 'MSK': 103
    },
    'fr':{
        'bert_model': 'camembert-base',
        'nationality': ['amérique', 'canada', 'japon', 'chine', 'corée', 'angleterre',
                        'france', 'allemagne', 'mexique', 'irak', 'irlande', 'iran',
                         'saudita', 'russie', 'vietnam', 'thaïlande', 'australie', 'espagne',
                         'turquie', 'israel', 'italie', 'egypte', 'somalie', 'india',
                         'brésil', 'colombie', 'grèce', 'afghanistan', 'cuba', 'syria'],
        'template_path': 'templates/templates_fr.txt',
        'occ_path': 'templates/occ_fr_revised.txt',
        'MSK': 104
    },
    'es':{
        # 'bert_model': 'bert-base-multilingual-uncased',
        'bert_model': 'dccuchile/bert-base-spanish-wwm-uncased',
        'nationality': ['américa', 'canadá', 'japón', 'china', 'corea', 'inglaterra',
                        'francia', 'alemania', 'méxico', 'irak', 'irlanda', 'irán', 'arabia',
                        'rusia', 'vietnam', 'tailandia', 'australia', 'españa', 'turquía',
                        'israel', 'italia', 'egipto', 'somalia', 'india', 'brasil',
                        'colombia', 'grecia', 'afganistán', 'cuba', 'siria', 'corea del norte', 'Pakistán', 'Rumania', 'Suiza', 'Marruecos'],
        'template_path': 'templates/templates_es.txt',
        'occ_path': 'templates/occ_es.txt',
        'MSK': 0
        # 'MSK': 103
    },
    'zh':{
        'bert_model': 'bert-base-chinese',
        # 'bert_model': 'bert-base-multilingual-uncased',
        'nationality': ['美国','加拿大','日本','中国','韩国','英格兰','法国','德国','墨西哥','伊拉克','爱尔兰',
                         '伊朗','沙特','俄国','越南','泰国','澳大利亚','西班牙','土耳其','以色列', '意大利','埃及',
                         '索马里','印度','巴西','哥伦比亚','希腊','阿富汗','古巴','叙利亚' ,'北朝鲜', '巴基斯坦', '罗马尼亚', '瑞士', '摩洛哥'],
        'template_path': 'templates/templates_zh.txt',
        'occ_path': 'templates/occ_zh.txt',
        'MSK': 103
        # 'MSK': 103
    },
    'jp':{
        # 'bert_model': 'cl-tohoku/bert-base-japanese',
        'bert_model': 'bert-base-multilingual-uncased',
        'nationality': ['アメリカ','カナダ','日本','中国','韓国','イギリス','フランス','ドイツ','メキシコ','イラク','アイルランド','イラン',
                        'サウジアラビア','ロシア','ベトナム','タイ','オーストラリア','スペイン','トルコ','イスラエル','イタリア','エジプト','ソマリア',
                        'インド','ブラジル','コロンビア','ギリシャ','アフガニスタン','キューバ','シリア'],
        'template_path': 'templates/templates_jp.txt',
        'occ_path': 'templates/occ_jp_revised.txt',
        # 'MSK': 4
        'MSK': 103
    },
    'tr':{
            'bert_model': 'dbmdz/bert-base-turkish-uncased',
            # 'bert_model': 'bert-base-multilingual-uncased',
            'nationality': ['Amerika', 'Kanada', 'Japonya', 'Çin', 'Kore', 'İngiltere',
                        'Fransa', 'Almanya', 'Meksika', 'Irak', 'İrlanda', 'İran', 'Suudi',
                        'Rusya', 'Vietnam', 'Tayland', 'Avustralya', 'İspanya', 'türkiye',
                        'İsrail', 'İtalya', 'Mısır', 'Somali', 'Hindistan', 'Brezilya',
                        'Kolombiya', 'Yunanistan', 'Afganistan', 'Küba', 'Suriye', 'Kuzey Kore', 'Pakistan', 'Romanya', 'İsviçre', 'Fas'],
            'template_path': 'templates/templates_tr.txt',
            'occ_path': 'templates/occ_tr.txt',
            # 'MSK': 103
            'MSK': 4
        },
    'ar':{
        'bert_model': 'asafaya/bert-base-arabic',
        # 'bert_model': 'bert-base-multilingual-uncased',
        'nationality': ['أمريكا', 'كندا', 'اليابان', 'الصين', 'كوريا', 'إنجلترا',
                        'فرنسا', 'ألمانيا', 'المكسيك', 'العراق', 'أيرلندا', 'إيران',
                         'السعودية', 'روسيا', 'فيتنام', 'تايلاند', 'أستراليا', 'أسبانيا',
                         'تركيا', 'إسرائيل', 'إيطاليا', 'مصر', 'الصومال', 'الهند',
                         'البرازيل', 'كولومبيا', 'اليونان', 'أفغانستان', 'كوبا', 'سوريا'],
        'template_path': 'templates/templates_ar.txt',
        'occ_path': 'templates/occ_ar.txt',
        'MSK': 4,
        # 'MSK': 103
    },
    'el':{
        'bert_model': 'nlpaueb/bert-base-greek-uncased-v1',
        # 'bert_model': 'bert-base-multilingual-uncased',
        'nationality': ["ΗΠΑ", "Καναδάς", "Ιαπωνία", "Κίνα", "Κορέα", "Αγγλία", "Γαλλία", "Γερμανία", "Μεξικό",
                        "Ιράκ", "Ιρλανδία", "Ιράν", "Σαουδική Αραβία", "Ρωσία","Βιετνάμ","Ταϊλάνδη","Αυστραλία",
                        "Ισπανία","Τουρκία","Ισραήλ","Ιταλία","Αίγυπτος","Σομαλία","Ινδία","Βραζιλία","Κολομβία",
                        "Ελλάδα","Αφγανιστάν","Κούβα","Συρία"],
        'template_path': 'templates/templates_el.txt',
        'occ_path': 'templates/occ_el.txt',
        'MSK': 103
    },
    'th':{
        'bert_model': 'monsoon-nlp/bert-base-thai',
        'nationality': ['อเมริกา', 'แคนาดา', 'ญี่ปุ่น', 'จีน', 'เกาหลี', 'อังกฤษ',
                         'ฝรั่งเศส', 'เยอรมนี', 'เม็กซิโก', 'อิรัก', 'ไอร์แลนด์', 'อิหร่าน', 'ซาอุดิ',
                         'รัสเซีย', 'เวียดนาม', 'ไทย', 'ออสเตรเลีย', 'สเปน', 'ตุรกี',
                         'อิสราเอล', 'อิตาลี', 'อียิปต์', 'โซมาเลีย', 'อินเดีย', 'บราซิล',
                         'โคลัมเบีย', 'กรีซ', 'อัฟกานิสถาน', 'คิวบา', 'ซีเรีย'],
        'template_path': 'templates/templates_th.txt',
        'occ_path': 'templates/occ_th.txt',
        'MSK': 104 #
    },
    'vi':{
        'bert_model': 'trituenhantaoio/bert-base-vietnamese-uncased',
        'nationality': ['Hoa Kỳ', 'canada', 'Nhật Bản', 'Đồ sứ', 'Hàn Quốc', 'Nước anh',
                        'Pháp', 'Đức', 'Mexico', 'Iraq', 'Ireland', 'iran',
                         'saudita', 'Nga', 'Việt Nam', 'thái lan', 'Châu Úc', 'Tây Ban Nha',
                         'Thổ nhĩ kỳ', 'israel', 'Ý', 'ai cập', 'somalia', 'Ấn Độ',
                         'brazil', 'colombia', 'hy lạp', 'afghanistan', 'cuba', 'syria'],
        'template_path': 'templates/templates_vi.txt',
        'occ_path': 'templates/occ_vi.txt',
        'MSK': 4 #
    },
}
