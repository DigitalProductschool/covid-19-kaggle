# covid-19-kaggle Documentation

## Our team:
We are an international team of passionate data scientist and AI engineers at Digital Product School at the Center for Business Creation and Innovation at Technical University of Munich, Germany. [Afsane Asaei](https://github.com/afiDPS), [Anubhav Jain](https://github.com/anubhav1997), and [Diana Amiri](https://github.com/dian-ai)

## Goal:
By participating in this challenge, we hope to contribute to this global effort against Covid19 pandemic. Our aim is to provide valuable insights regarding the questions of [Task 3](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=567) for the doctors and researchers. We plan to do text data mining on scientific literatures. This method has been proven to be a good strategy to handle fast growing accumulation of scientific papers, which is a valid case at this moment due to the world-wide pandemic situation of Covid-19. Therefore, a sequence of steps needs to be done to find the most relative articles and snippets to the corresponding key questions. Our general approach includes feature engineering with TFIDF and PCA, topic modeling with unsupervised learning, and text classification with KNN to ease the process of finding the most relative answer.

## [Dataset description](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge):
In response to the COVID-19 pandemic, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 52,000 scholarly articles, including over 41,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease. There is a growing urgency for these approaches because of the rapid acceleration in new coronavirus literature, making it difficult for the medical research community to keep up.

## Our Approach:
* For data analysing
  - Defining influential factors based on specific features of a defined task is crucial as data grows and the complexity increases. Hence our group decided to label the provided large dataset based on the **language** of the article, if the article is related to **COVID-19**, **genomic sequence** of virus, and whether it has discussions on **receptor surface of proteins** of target virus. We believe this labeling helps us to narrow down the noise in the dataset, which means that we drop the articles that are not in this criteria, and it affects the performance of the classifier.
  <br/>
  
  
* For pre-processing:
  - Now that our data is subject-oriented, we need to remove irrelevent terms, symbols, and punctuation marks.
  
* For feature engineering:
  - The method used for converting the preprocessed text data into feature vectors is called feature engineering. For this section, we have used four different methods to covert full body texts into feature vectors.
**TF-IDF**, **Doc2Vec**, **Doc2Vec and TF-IDF together**, **Bigram phrase modeling and Doc2Vec**.
Afterwards the PCA was implemented on feature vectors to reduce the dimensionality to 2 so that we can visualize the articles in space.

* For visualization and evaluation:
  - Clustering (unsupervised learning method):
    K-means clustering is an interesting way of finding patterns regarding similarity of articles based on their content. With the help of elbow method we found the optimal numbers of clustering.
    <br/>


<p align="center">
  <img src="/Assets/covid19_elbow(1).png"  width="250" height="250" title = "elbow method, K=10">
</p>
<p align="center">
  <img src="/Assets/covid19_label_TFIDF(1).png" width="250" height="250" title = "TFIDF">
  <img src="/Assets/covid19_label_Doc2Vec.png" width="250" height="250" title = "Doc2Vec">
  <img src="/Assets/covid19_label_Doc2Vec_TFIDF.png" width="250" height="250" title = "TFIDF & Doc2Vec"> 
  <img src="/Assets/dmm_dbow_pca_k10.png" width="222" height="218" title = "Bigram & Doc2Vec (dmm +dbow)">
 </p>

  - To quantitatively measure the feature vectors obtained from different algorithms, the S. Vajda et al. [5](https://www.researchgate.net/publication/316550769_A_Fast_k-Nearest_Neighbor_Classifier_Using_Unsupervised_Clustering) proposes an algorithm which can be briefed as follow. Firstly we cluster all the feature vectors thus labels for all the feature vectors are obtained. Now our current labeled data can be splited into training and testing sets. Thereafter, we perform K-Nearest Neighbour classification of the test set using the labels of the training set and messure the accuracy of KNN. The results of different algorithms have been compared in the bar graph. It was observed that TF-IDF vectorization performed better than the other algorithms.
  <br/>
<p align="center">
  <img src="/Assets/Comparison of different algorithms(1).png"  width="300" height="300">
</p>

  * For finding answers:
    - To find the most relevant documents to the answers requires finding a good similarity measure. Therefore the task description was pre-processed as same as body text documents, then we projected the task description to the same feature space as the body text documents (pre-trained TF-IDF model and pre-trained PCA model).
For comparing documents cosine similarity was used. This is a better choice because even if a document is far apart in the euclidean distance (due to the size of the document) it could still be oriented closer.
We have also formed word clouds of the top 20 most relevant documents to see the top keywords.

* Supervised Qualitative Analysis of Results:
  - A supervised evaluation was conducted to better judge and compare the performance of the algorithms. Dr. Hassan Vahidnezhad evaluated the word clouds along with the looking at the table containing the top 10 most similar documents. Based on his expert opinion, TF-IDF performed better than other algorithms based on the table and keywords that the doctor observed.
  
  
  
  

### Documents relevant to general task 3 description




|    | title  | abstract  | authors  | journal | body-text   | cosine-similarity  |   |   |   |
|---|---|---|---|---|---|---|---|---|---|
| 26261  | Presumptive meningoencephalitis secondary to extension of otitis mediainterna caused by Streptococcus equi subspecies zooepidemicus in a cat  |   |  Martin-Vaquero, Paula; da Costa, Ronaldo C.; Daniels, Joshua B.  | Journal of Feline Medicine & Surgery | A 5-year-old castrated male domestic longhair cat was presented with neurological signs consistent with a central vestibular lesion and left Horner's syndrome. Computed tomography images revealed hyperattenuating, moderately contrast-enhancing material within the left tympanic bulla, most consis | 1.0  |   |   |   |   |
| 18620  | Conservation of nucleotide sequences for molecular diagnosis of Middle East respiratory syndrome coronavirus, 2015  |   | Furuse, Yuki; Okamoto, Michiko; Oshitani, Hitoshi  | International Journal of Infectious Diseases  | Middle East respiratory syndrome coronavirus (MERS-CoV) is an enveloped virus with a positive-sense RNA genome. Infection with the virus causes severe respiratory symptoms in humans, with a case fatality rate as high as 37%. 1 Camels may be a source of infection to humans. 2 Human-to-human trans  | 1.0  |   |   |   |
| 18448  | Chapter 68 Acute Colitis in Horses  |   | McConnico, Rebecca S. | Robinson's Current Therapy in Equine Medicine  |A cute colitis is a common cause of rapid debilitation and death in horses. More than 90% of untreated horses with this condition die or are euthanized, but horses that are treated appropriately usually respond and gradually recover over a 7-to 14-day period. Colitis-associated diarrhea is spora   |  1.0 |   |   |   |
| 34891  | Genotyping of clinically relevant human adenoviruses by array-in-well hybridization assay  | A robust oligonucleotide array-in-well hybridization assay using novel up-converting phosphor reporter technology was applied for genotyping clinically relevant human adenovirus types. A total of 231 adenovirus-positive respiratory, ocular swab, stool and other specimens from 219 patients collec  | Ylihärsilä, M.; Harju, E.; Arppe, R.; Hattara, L.; Hölsä, J.; Saviranta, P.; Soukka, T.; Waris, M.  | Clinical Microbiology and Infection  | Human adenoviruses (hAdVs) belong to the genus Mastadenovirus of the family Adenoviridae. There are presently 57 different hAdV types grouped into seven species A to G [1] . Different types cause a wide range of acute and chronic diseases. Acute respiratory disease is predominantly caused by the  |   1.0|   |   |   |
