# aws-mls-notes
My notes  for AWS machine learning specialist certificate exam

## Amazon SageMaker
* SageMaker GroundTruth – a data labeling service that lets you use workforce (human annotators) through your own private annotators, Amazon Mechanical Turk, or third-party services
* In SageMaker, with Pipe input mode, your data is fed on-the-fly into the algorithm container without involving any disk I/O. This approach shortens the lengthy download process and dramatically reduces startup time
* You MUST use protobuf RecordIO as your training data format before you can take advantage of the Pipe mode.
* An Amazon SageMaker managed VPC can only be created in an Amazon managed Account. Notebooks can run inside AWS managed VPC or customer managed VPC.
* The Amazon SageMaker BlazingText algorithm provides highly optimized implementations of the Word2vec and text classification algorithms
* Using the RecordIO protobuf format is a best practice for preparing data for use with Amazon SageMaker, and it is specifically recommended for use with the built-in algorithms.
* The Amazon SageMaker DeepAR forecasting algorithm is a supervised learning algorithm for forecasting scalar (one-dimensional) time series using recurrent neural networks (RNN)
* SageMakerVariantInvocationsPerInstance = (MAX_RPS * SAFETY_FACTOR) * 60
* The Amazon SageMaker Latent Dirichlet Allocation (LDA) algorithm is an unsupervised learning algorithm that attempts to describe a set of observations as a mixtureof distinct categories
* Amazon SageMaker Autopilot is a feature-set that simplifies and accelerates various stages of the machine learning workflow by automating the process of buildingand deploying machine learning models (AutoML).
* By stopping and restarting the SageMaker notebook instance, it will automatically apply the latest security and software updates provided by SageMaker.
* SageMaker Data Wrangler – a visual data preparation and cleaning tool that allows data scientists and engineers to easily clean and prepare data for machine learning.
* SageMaker Neo – allows you to optimize machine learning models for deployment on edge devices to run faster with no loss in accuracy.
* Managed Spot Training – allows data scientists and engineers to save up to 90% on the cost of training machine learning models by using spare compute capacity.
* Amazon CloudTrail helps you detect unauthorized SageMaker API calls.
* Multi-model endpoints provide a scalable and cost-effective solution to deploying large numbers of models. They use the same fleet of resources and a shared serving container to host all of your models
* Amazon SageMaker Object2Vec generalizes the Word2Vec embedding technique for words to more complex objects, such as sentences and paragraphs.

## Questions:

* Amazon DeepLen is a deep learning-enabled camera for developers
* Amazon Comprehend is managed Natural Language Processing (NLP) service, for entity recognition, sentiment analysis, language/key phrases detection, PII detection and syntax analysis
* Amazon rekognition is a service that makes it easy to add powerful visual analysis to your applications, for image and video object detectio, image search, face verification
* Latent Dirichlet Allocation (LDA)  is for topic modeling
* A given time series is thought to consist of three systematic components including level, trend, seasonality, and one non-systematic component called noise.These components are defined as follows:
    * Level: The average value in the series.
    * Trend: The increasing or decreasing value in the series.
    * Seasonality: The repeating short-term cycle in the series.
    * Noise: The random variation in the series.
* In regression,   Residual plot distribution indicates over or under-estimations
* You can use Amazon Polly to generate speech from either plain text or from documents marked up with Speech Synthesis Markup Language (SSML).
* Improver query performance of Athena
    * Compressing
    * Partitioning
    * Change to columnar format like Apache Parquet and Apache ORC
* Multiple imputation fills in missing values by generating plausible numbers derived from distributions of and relationships among observed variables in the data set.
* Random cut forest(RCF) is an unsupervised learning model. RCF is used for anomaly detection, not classification.
* Horovod is a distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet
* AWS Data Pipeline can be used to move the hourly data, as it provides a way to move data from various sources to Amazon EMR for processing
* Amazon Lex is a fully managed AI service for building conversational interfaces into any application. Start creating a chatbot, voice assistant
* Amazon Polly turns text into life like speech, allowing you to create applications that talk
* stratified k-fold cross-validation will enforce the class distribution in each split of the data to match the distribution in the complete training dataset
* Kinesis Data stream, Max. ingestion per shard = 1000 KB/s
* Amazon Elastic Inference allows you to attach low-cost GPU-powered acceleration to Amazon EC2 and Amazon SageMaker instances to run inference workloads with a fraction of the compute resources
* AWS Lake Formation provides machine learning capabilities to create custom transforms to cleanse your data. There is currently one available transform named FindMatches. The FindMatches transform enables you to identify duplicate or matching records in your dataset, even when the records do not have a common unique identifier and no fields match exactly
* Amazon Augmented AI (A2I) allows you to conduct a human review of machine learning (ML) systems to guarantee precision
* Amazon Textract is a machine learning (ML) service that uses optical character recognition (OCR) to automatically extract text, handwriting, and data
* Amazon FSx is a fully managed third-party file system solution. It uses SSD storage to provide fast performance with low latency
* Macie will search data to discover, monitor, and protect sensitive data. Managed data identifiers are pretrained to detect sensitive data types.Amazon Macie is a data security service that discovers sensitive data by using machine learning and pattern matching, provides visibility into data security risks,and enables automated protection against those risks.
* AWS Glue is more than a standard ETL tool. AWS Glue automatically classifies data, creates schemas, and catalogs metadata.
* AWS Panorama is a collection of machine learning (ML) devices and a software development kit (SDK) that brings CV to on-premises internet protocol (IP) cameras.
* Linear models are supervised learning algorithms used for solving either classification or regression problems.
* AWS Kinesis Firehose is primarily focused on efficiently loading streaming data into various destinations, while Kinesis Data Streams provides a more flexibleand customizable platform for building real-time streaming applications. Kinesis Data Analytics complements both services by providing an SQL-based approach to perform real-time analytics on streaming data.
* The Factorization Machines algorithm is a general-purpose supervised learning algorithm that you can use for both classification and regression tasks. It is an extension of a linear model that is designed to capture interactions between features within high dimensional sparse datasets economically. For example, in a click prediction system, the Factorization Machines model can capture click rate patterns observed when ads from a certain ad-category are placed on pages from a certain page-category. Factorization machines are a good choice for tasks dealing with high dimensional sparse datasets, such as click prediction and item recommendation.
* SMOTE is an oversampling technique that generates synthetic samples from the minority class


| Scenario    | Solution |
| -------- | ------- |
|**MLS-C01 Domain 1: Data Engineering** ||
|A company wants to automatically convert streaming JSON data into Apache Parquet before storing them in an S3 bucket|Use Amazon Kinesis Firehose|
|A company uses Amazon EMR for its ETL processes. The company is looking for an alternative with a lower operational overhead|Run the ETL jobs using AWS Glue|
|Which service should you use to deliver streaming data from Amazon MSK to a Redshift cluster with low latency?|Redshift Streaming Ingestion|
|A data engineer is building a pipeline for streaming data. The data will be fetched from various sources.|Create an application that uses Kinesis Producer Library (KPL) to load streaming data from various sources into a Kinesis Data stream.|
|A company wants to set up a data lake on Amazon S3. The data will be sourced from S3 buckets located in different AWS accounts. Which service can simplify the implementation of the data lake?|AWS Lake Formation|
|**MLS-C01 Domain 2: Exploratory Data Analysis**||
|An image classifier is getting high accuracy on the validation dataset. However, the accuracy significantly dropped when tested against real data. How can you improve the model’s performance?|Take existing images from the training data. Apply data augmentation techniques (ex: flipping, rotating, adjusting brightness) to the images and add them to the training data. Retrain the model|
|What methods can a machine learning engineer use to reduce the size of a large dataset while retaining only relevant features?|1. Principal Component Analysis (PCA) 2. t-Distributed Stochastic Neighbor Embedding (t-SNE)|
|A dataset contains a mixture of categorical and numerical features. What feature engineering method should be done to prepare the data for training?|One-hot encoding|
|X and Y variables have a correlation coefficient of -0.98. What does it indicate?|Very strong negative correlation |
|A machine learning engineer handles a small dataset with missing values. What should they do to ensure no data points are lost?|Use imputation techniques to fill in missing values|
|**MLS-C01 Domain 3: Modeling**||
|An ML engineer wants to evaluate the performance of a binary classification model visually. What visualization technique should be used?|Confusion matrix|
|An ML engineer wants to discover topics available within a large text dataset. Which algorithm should the engineer train the model on?|Latent Dirichlet Allocation (LDA) algorithm|
|A SageMaker Object2vec model is overfitting on a validation dataset. How do you solve this problem?|Use Regularization, in this case, adjusting the value of the Dropout parameter.|
|A neural network model is being trained using a large dataset in batches. As the training progresses, the loss function begins to oscillate. Which could be the cause?|The learning rate is too high|
|What SageMaker built-in algorithm is suitable for predicting click-through rate (CTR) patterns?|Factorization machines|
|**MLS-C01 Domain 4: Machine Learning Implementation and Operations**||
|An ML engineer wants to auto-scale the instances behind a SageMaker endpoint according to the volume of incoming requests. Which metric should this scaling be based on?|InvocationsPerInstance|
|Which AWS service can you use to convert audio formats into text?|Amazon Transcribe|
|An ML engineer is training a cluster of SageMaker instances. The traffic between the instances must be encrypted.|Enable inter-container traffic encryption|
|A company wants to use Amazon SageMaker to deploy various ML models in a cost-effective way.|Use multi-model endpoint|
|What AWS service can help you build an AI-powered chatbot that can interact with customers?|Amazon Lex|

Reference Notes: https://docs.google.com/document/d/158_KYe21vzQwI8nsu1ZNX2L27S6eLrzm9SeN5cQhlLs/edit



