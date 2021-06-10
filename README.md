# Tugas-uas
WEB MINING(resume)

Nama  : Adam Tio Prayogi

Nim   : 180411100094

# Materi 1 (Crawling Data)

   -Crawling Data adalahadalah teknik pengumpulan data yang digunakan untuk mengindeks informasi pada
   halaman menggunakan URL dengan menyertakan API  untuk melakukan penambangan dataset yang lebih besar.
   
   ^Crawling Menggunakan Python
                
      pip install tweepy  #installasi
      
      import tweepy #impotr_modul
      import csv
      
      access_token="" #variabel_penampungdata
      access_token_secret=""
      consumer_key=""
      consumer_key_secret=""
      
      access_token=""
      access_token_secret=""
      consumer_key=""
      consumer_key_secret=""
      
      csvFile = open('nama-file.csv', 'w', encoding='utf-8')
      csvWriter = csv.writer(csvFile)
      
      for tweet in tweepy.Cursor(api.search, q='#Python -filter:retweets', tweet_mode='extended',lang="id", since='2021-01-01', until='2021-01-10').items(100):
        text = tweet.full_text
        user = tweet.user.name
        created = tweet.created_at
        csvWriter.writerow([created, text.encode('utf-8'), user])
    csvWriter = csv.writer(csvFile)
    csvFile.close()

# materi 2 (Pre-Processing Data)
  
 Pada proses Pre-Processing data ini meliputi beberapa tahaban yaitu:

  (1) case folding, 

  (2) tokenizing,

  (3) filtering,

  (4) stemming.
  
     import pandas as pd 
     import numpy as np
     TWEET_DATA = pd.read_csv("dataset_sementara.csv")
     TWEET_DATA.head()
     
   >Case Folding adalah tahap untuk konversi text menjadi suatu bentuk yang standar
    
     # gunakan fungsi Series.str.lower() pada Pandas
     TWEET_DATA['tweet'] = TWEET_DATA['tweet'].str.lower()

     print('Case Folding Result : \n')
     print(TWEET_DATA['tweet'].head(5))
     print('\n\n\n')
     
      import string 
      import re #regex library

      # import word_tokenize & FreqDist from NLTK
      from nltk.tokenize import word_tokenize 
      from nltk.probability import FreqDist

   >Tokenizing, merupakan tahapan pemotongan string inputan berdasarkan pada tipe kata yang menyusunya.

      def remove_tweet_special(text):
          # remove tab, new line, ans back slice
          text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
          # remove non ASCII (emoticon, chinese word, .etc)
          text = text.encode('ascii', 'replace').decode('ascii')
          # remove mention, link, hashtag
          text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
          # remove incomplete URL
          return text.replace("http://", " ").replace("https://", " ")

      TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_tweet_special)

      #remove number
      def remove_number(text):
          return  re.sub(r"\d+", "", text)

      TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_number)

      #remove punctuation
      def remove_punctuation(text):
          return text.translate(str.maketrans("","",string.punctuation))

      TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_punctuation)

      #remove whitespace leading & trailing
      def remove_whitespace_LT(text):
          return text.strip()

      TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_whitespace_LT)

      #remove multiple whitespace into single whitespace
      def remove_whitespace_multiple(text):
          return re.sub('\s+',' ',text)

      TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_whitespace_multiple)

      # remove single char
      def remove_singl_char(text):
          return re.sub(r"\b[a-zA-Z]\b", "", text)

      TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_singl_char)

      # NLTK word rokenize 
      def word_tokenize_wrapper(text):
          return word_tokenize(text)

      TWEET_DATA['tweet_tokens'] = TWEET_DATA['tweet'].apply(word_tokenize_wrapper)

      print('Tokenizing Result : \n') 
      print(TWEET_DATA['tweet_tokens'].head())
      print('\n\n\n')
      
   >filtering, bertujuan untuk mengambil kata-kata penting pada tokens yang dihasilkan oleh proses sebelumnya
      
      from nltk.corpus import stopwords

      # ----------------------- get stopword from NLTK stopword -------------------------------
      # get stopword indonesia
      list_stopwords = stopwords.words('indonesian')


      # ---------------------------- manualy add stopword  ------------------------------------
      # append additional stopword
      list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                             'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                             'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                             'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                             'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                             'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                             '&amp', 'yah'])

      # ----------------------- add stopword from txt file ------------------------------------
      # read txt stopword using pandas
      txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)

      # convert stopword string to list & append additional stopword
      list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

      # ---------------------------------------------------------------------------------------

      # convert list to dictionary
      list_stopwords = set(list_stopwords)


      #remove stopword pada list token
      def stopwords_removal(words):
          return [word for word in words if word not in list_stopwords]

      TWEET_DATA['tweet_tokens_WSW'] = TWEET_DATA['tweet_tokens'].apply(stopwords_removal) 


      print(TWEET_DATA['tweet_tokens_WSW'].head())
      
   >Steemer, Ditahap ini kita akan .apply() function stemmer dari library Sastrawi untuk mengembalikan kata kebentuk dasarnya.
      
      conda install -c conda-forge swifter
      
      # import Sastrawi package
      from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
      import swifter


      # create stemmer
      factory = StemmerFactory()
      stemmer = factory.create_stemmer()

      # stemmed
      def stemmed_wrapper(term):
          return stemmer.stem(term)

      term_dict = {}

      for document in TWEET_DATA['tweet_normalized']:
          for term in document:
              if term not in term_dict:
                  term_dict[term] = ' '

      print(len(term_dict))
      print("------------------------")

      for term in term_dict:
          term_dict[term] = stemmed_wrapper(term)
          print(term,":" ,term_dict[term])

      print(term_dict)
      print("------------------------")


      # apply stemmed term to dataframe
      def get_stemmed_term(document):
          return [term_dict[term] for term in document]

      TWEET_DATA['tweet_tokens_stemmed'] = TWEET_DATA['tweet_normalized'].swifter.apply(get_stemmed_term)
      print(TWEET_DATA['tweet_tokens_stemmed'])
      
      TWEET_DATA.to_csv("Text_Preprocessing.csv") #savekecsv
      TWEET_DATA.to_excel("Text_Preprocessing.xlsx") #saveke excel
      
      
# materi 3 (TF-IDF/Term Frequency-Inverse Document Frequency)

TF-IDF adalah ukuran statistik yang menggambarkan pentingnya suatu istilah terhadap sebuah 
dokumen dalam sebuah kumpulan atau korpus 1. ukuran ini sering digunakan sebagai
faktor pembobot dalam pencarian temu balik informasi (information retrieval), 
penambangan text (text mining), dan pemodelan pengguna (user-modeling). Dalam TF-IDF Term
frequency menyatakan fekuensi (tingkat keseringan) munculnya seuatu term dalam dokumen, 
sedangkan document frequency merupakan banyaknya jumlah dokumen dimana sebuah term itu
muncul2.

> Siapakan data yang akan diolah, baca file menggunakan Pandas. 
  Kolom yang dibutuhkan dalam pemrosesan kali ini adalah kolom label dan tweet_tokens_stemmed . 
  Kemudian rename kolom pada Dataframe menjadi ["label", "tweet"] .

    import pandas as pd 
    import numpy as np

    TWEET_DATA = pd.read_csv("Text_Preprocessing.csv", usecols=["label", "tweet_tokens_stemmed"])
    TWEET_DATA.columns = ["label", "tweet"]

    TWEET_DATA.head()
    
    # convert list formated string to list
    
    import ast

    def convert_text_list(texts):
        texts = ast.literal_eval(texts)
        return [text for text in texts]

    TWEET_DATA["tweet_list"] = TWEET_DATA["tweet"].apply(convert_text_list)


    print(TWEET_DATA["tweet_list"][90])

    print("\ntype : ", type(TWEET_DATA["tweet_list"][90]))
    
  
    # convert list formated string to list Jika ketika kita cek, item di kolom tweet memiliki tipe data <class str> , 
    # maka kita perlu mengubah string menjadi list dan menyimpan pada Pandas Seriec tweet_list ,
    
    import ast

    def convert_text_list(texts):
      texts = ast.literal_eval(texts)
      return [text for text in texts]

    TWEET_DATA["tweet_list"] = TWEET_DATA["tweet"].apply(convert_text_list)


    print(TWEET_DATA["tweet_list"][90])

    print("\ntype : ", type(TWEET_DATA["tweet_list"][90]))
    
  >Term Frequency (TF)
  
    def calc_TF(document):
        # Counts the number of times the word appears in review
        TF_dict = {}
        for term in document:
            if term in TF_dict:
                TF_dict[term] += 1
            else:
                TF_dict[term] = 1
        # Computes tf for each word
        for term in TF_dict:
            TF_dict[term] = TF_dict[term] / len(document)
        return TF_dict

    TWEET_DATA["TF_dict"] = TWEET_DATA['tweet_list'].apply(calc_TF)

    TWEET_DATA["TF_dict"].head()
    # Check TF result
    index = 90

    print('%20s' % "term", "\t", "TF\n")
    for key in TWEET_DATA["TF_dict"][index]:
        print('%20s' % key, "\t", TWEET_DATA["TF_dict"][index][key])
    
   >Inverse Document Frequency (IDF)
   
     def calc_DF(tfDict):
      count_DF = {}
      # Run through each document's tf dictionary and increment countDict's (term, doc) pair
      for document in tfDict:
          for term in document:
              if term in count_DF:
                  count_DF[term] += 1
              else:
                  count_DF[term] = 1
      return count_DF

    DF = calc_DF(TWEET_DATA["TF_dict"])
    
    n_document = len(TWEET_DATA)

    def calc_IDF(__n_document, __DF):
        IDF_Dict = {}
        for term in __DF:
            IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
        return IDF_Dict

    #Stores the idf dictionary
    IDF = calc_IDF(n_document, DF)
    
    #calc TF-IDF
    
    def calc_TF_IDF(TF):
        TF_IDF_Dict = {}
        #For each word in the review, we multiply its tf and its idf.
        for key in TF:
            TF_IDF_Dict[key] = TF[key] * IDF[key]
        return TF_IDF_Dict

    #Stores the TF-IDF Series
    TWEET_DATA["TF-IDF_dict"] = TWEET_DATA["TF_dict"].apply(calc_TF_IDF)
    
    # Check TF-IDF result
    index = 90

    print('%20s' % "term", "\t", '%10s' % "TF", "\t", '%20s' % "TF-IDF\n")
    for key in TWEET_DATA["TF-IDF_dict"][index]:
        print('%20s' % key, "\t", TWEET_DATA["TF_dict"][index][key] ,"\t" , TWEET_DATA["TF-IDF_dict"][index][key])
        
      # sort descending by value for DF dictionary 
      sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)[:50]

      # Create a list of unique words from sorted dictionay `sorted_DF`
      unique_term = [item[0] for item in sorted_DF]

      def calc_TF_IDF_Vec(__TF_IDF_Dict):
          TF_IDF_vector = [0.0] * len(unique_term)

          # For each unique word, if it is in the review, store its TF-IDF value.
          for i, term in enumerate(unique_term):
              if term in __TF_IDF_Dict:
                  TF_IDF_vector[i] = __TF_IDF_Dict[term]
          return TF_IDF_vector

      TWEET_DATA["TF_IDF_Vec"] = TWEET_DATA["TF-IDF_dict"].apply(calc_TF_IDF_Vec)

      print("print first row matrix TF_IDF_Vec Series\n")
      print(TWEET_DATA["TF_IDF_Vec"][0])

      print("\nmatrix size : ", len(TWEET_DATA["TF_IDF_Vec"][0]))
      
      # Convert Series to List
      
      TF_IDF_Vec_List = np.array(TWEET_DATA["TF_IDF_Vec"].to_list())

      # Sum element vector in axis=0 
      sums = TF_IDF_Vec_List.sum(axis=0)

      data = []

      for col, term in enumerate(unique_term):
          data.append((term, sums[col]))

      ranking = pd.DataFrame(data, columns=['term', 'rank'])
      ranking.sort_values('rank', ascending=False)
      
      TWEET_DATA.to_csv("Text_TF-IDF.csv")
      
      TWEET_DATA.to_excel("Text_TF-IDF.xlsx")
      
      TWEET_DATA.to_hdf("Text_TF-IDF.h5")
      
# materi 4 (modeling data)
Pemodelan pengguna (User Modeling) merupakan subdivisi dari interkaso manusia dan komputer 
yang menggambarkan proses membangun dan memodifikasi pemahaman konseptual pengguna dimana 
tujuan dari pemodelan ini adalah penyesuaian dan adaptasi sistem dengan kebutuhan spesifik 
dari pengguna.

- K-NN (K-Nearest Neighbors)¶
Pada penerapanya kali ini kita menggunakan pemodelan data menggunakan klasifikasi KNN (
K-Nearest Neighbors). klasifikasi ini merupakan sebuah non-parametric dan lazy learning 
algorithm3. Non-parametric berarti tidak ada asumsi untuk distribusi data yang mendasarinya, 
dengan demikian struktur model ditentukan oleh dataset. Sedangkan Lazy Algorithm / algoritma
malas diartikan tidak memerlukan titik data latih untuk pembuatan model. Akibat dari ini
menimbulkan maslah dari fase pengujian dimana dalam fase tersebut lebih lambat dan lebih mahal,
dalam artian waktu dan memory. Dalam Algoritma KNN, K adalah jumlah tetangga terdekat, 
dimana jumlah tetangga dalah faktor penentu inti. K umumnya merupakan bilangan ganjil 
jika jumlah kelas adalah 2, dan bila K = 1 maka algoritma tersebut dikenal dengan nama 
algoritma tetangga terdekat. Algoritma K-NN memiliki tahaban sebagai berikut :

> 1. Menghitung Jarak
> 2. Menentukan tetangga terdekat
> 3. Memberi suara untuk label

>Load Data Set

    import pandas as pd 
    import numpy as np

    #Import scikit-learn dataset library
    from sklearn import datasets

    #Load dataset
    wine = pd.read_csv("TF-IDF.csv")

    wine.head()
    
>Memisahkan Data

    # Import train_test_split function
    from sklearn.model_selection import train_test_split

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) # 70% training and 30% test
    
 >Model Untuk K = 5
  
    #Import knearest neighbors Classifier model
    from sklearn.neighbors import KNeighborsClassifier

    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    #Train the model using the training sets
    knn.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = knn.predict(X_test)
    
  >evaluasi
  
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
