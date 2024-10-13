import os
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_sgm_folder(folder_path):
    """
    Extracts title and body text from all .sgm files in the specified folder.
    """
    text_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.sgm'):
            file_path = os.path.join(folder_path, file_name)
            text_data += extract_text_from_sgm(file_path)
    return text_data

def extract_text_from_sgm(file_path):
    """
    Extracts text from a single .sgm file.
    """
    with open(file_path, 'r', encoding='latin-1') as file:
        soup = BeautifulSoup(file, 'html.parser')

        # Extract articles and headlines
        articles = soup.find_all('reuters')
        text_data = []

        for article in articles:
            title = article.find('title')
            body = article.find('body')

            if title and body:
                title_text = title.get_text().strip()
                body_text = body.get_text().strip()
                text_data.append((title_text, body_text))
        return text_data

def tokenize_text(text):
    """
    Tokenizes and cleans text by removing stop words and non-alphabetic tokens.
    """
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]  # Remove non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

# Example of extracting text from the Reuters21578 dataset
folder_path = '/Users/sujithkumaravel/Downloads/reuters21578'
text_data = extract_text_from_sgm_folder(folder_path)

# Tokenize the extracted text data
token_stream = [tokenize_text(title + ' ' + body) for title, body in text_data]

def spimi_index(token_stream):
    """
    Creates a simple SPIMI-inspired inverted index where tokens map to document IDs.
    """
    index = {}
    doc_id = 0  # Use doc_id to track documents

    for tokens in token_stream:
        doc_id += 1
        for token in tokens:
            if token in index:
                index[token].add(doc_id)
            else:
                index[token] = {doc_id}
    return index

def create_inverted_index(token_stream):
    """
    Creates an inverted index mapping each token to the list of document IDs where it appears.
    """
    inverted_index = {}
    for doc_id, tokens in enumerate(token_stream):
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append(doc_id)
    return inverted_index

def create_positional_index(token_stream):
    """
    Creates a positional inverted index where tokens are mapped to their positions within documents.
    """
    positional_index = {}
    for doc_id, tokens in enumerate(token_stream):
        for pos, token in enumerate(tokens):
            if token not in positional_index:
                positional_index[token] = {}
            if doc_id not in positional_index[token]:
                positional_index[token][doc_id] = []
            positional_index[token][doc_id].append(pos)
    return positional_index

# Build inverted index and positional index
inverted_index = create_inverted_index(token_stream)
positional_index = create_positional_index(token_stream)

def boolean_and(postings1, postings2):
    """
    Performs an AND operation between two sets of postings (intersection).
    """
    return postings1.intersection(postings2)

def boolean_or(postings1, postings2):
    """
    Performs an OR operation between two sets of postings (union).
    """
    return postings1.union(postings2)

def near_operator(term1, term2, k, positional_index):
    """
    Implements the NEAR operator to find documents where term1 and term2 appear within k tokens of each other.
    """
    result_docs = []
    if term1 in positional_index and term2 in positional_index:
        for doc_id in positional_index[term1]:
            if doc_id in positional_index[term2]:
                positions1 = positional_index[term1][doc_id]
                positions2 = positional_index[term2][doc_id]
                for pos1 in positions1:
                    for pos2 in positions2:
                        if abs(pos1 - pos2) <= k:
                            result_docs.append(doc_id)
    return result_docs

def concordance(query, k, token_stream, positional_index):
    """
    Implements the CONCORDANCE function which returns occurrences of the query term with k tokens of context.
    """
    result = []
    if query in positional_index:
        for doc_id, positions in positional_index[query].items():
            for position in positions:
                left_context = token_stream[doc_id][max(0, position-k):position]
                right_context = token_stream[doc_id][position+1:min(len(token_stream[doc_id]), position+k+1)]
                result.append(f"{doc_id}: {' '.join(left_context)} {query} {' '.join(right_context)}")
    return result

def calculate_statistics(token_stream):
    """
    Calculates statistical properties of terms in the token stream and returns a DataFrame.
    """
    term_stats = []
    total_tokens = 0

    for tokens in token_stream:
        total_tokens += len(tokens)

    # Count terms and postings at various preprocessing steps
    num_unfiltered_terms = len(set(token for tokens in token_stream for token in tokens))
    num_nonpositional_postings = total_tokens

    # Simulate preprocessing steps
    for step in [('No Numbers', 'no_numbers'),
                 ('Case Folding', 'case_folding'),
                 ('30 Stop Words', '30_stop_words'),
                 ('150 Stop Words', '150_stop_words'),
                 ('Stemming', 'stemming')]:

        step_name, step_method = step
        if step_method == 'no_numbers':
            filtered_tokens = [t for t in tokens if t.isalpha()]
        elif step_method == 'case_folding':
            filtered_tokens = [t.lower() for t in tokens if t.isalpha()]
        elif step_method == '30_stop_words':
            filtered_tokens = [t for t in tokens if t.isalpha() and t not in set(stopwords.words('english'))][:30]
        elif step_method == '150_stop_words':
            filtered_tokens = [t for t in tokens if t.isalpha() and t not in set(stopwords.words('english'))][:150]
        elif step_method == 'stemming':
            from nltk.stem import PorterStemmer
            ps = PorterStemmer()
            filtered_tokens = [ps.stem(t) for t in tokens if t.isalpha()]

        num_terms = len(set(filtered_tokens))
        num_postings = len(filtered_tokens)

        delta_percent_terms = ((num_unfiltered_terms - num_terms) / num_unfiltered_terms) * 100 if num_unfiltered_terms > 0 else 0
        delta_percent_postings = ((num_nonpositional_postings - num_postings) / num_nonpositional_postings) * 100 if num_nonpositional_postings > 0 else 0

        term_stats.append((step_name, num_terms, delta_percent_terms, num_postings, delta_percent_postings))

    # Convert to DataFrame for easy visualization
    df_stats = pd.DataFrame(term_stats, columns=['Preprocessing Step', 'Num Terms', '∆% Terms', 'Num Nonpositional Postings', '∆% Postings'])
    df_stats['T% Terms'] = ((df_stats['Num Terms'].cumsum() / num_unfiltered_terms) * 100).round(2)
    df_stats['T% Postings'] = ((df_stats['Num Nonpositional Postings'].cumsum() / num_nonpositional_postings) * 100).round(2)

    return df_stats

# Example queries
term1 = "bush"
term2 = "reagan"
k = 5

# Example Boolean AND, OR, and NEAR queries
docs_and = boolean_and(set(inverted_index.get(term1, [])), set(inverted_index.get(term2, [])))
docs_or = boolean_or(set(inverted_index.get(term1, [])), set(inverted_index.get(term2, [])))
docs_near = near_operator(term1, term2, k, positional_index)

# Example Concordance
concordance_result = concordance("climate", 10, token_stream, positional_index)

def main():
    # Phase 0: Extract and tokenize text from Reuters21578 dataset
    folder_path = '/Users/sujithkumaravel/Downloads/reuters21578'  # Update to your Reuters21578 dataset path
    print("Extracting text from Reuters21578 dataset...")
    text_data = extract_text_from_sgm_folder(folder_path)

    print("Tokenizing text...")
    token_stream = [tokenize_text(title + ' ' + body) for title, body in text_data]

    # Phase 1: Build the indexes
    print("Building inverted index...")
    inverted_index = create_inverted_index(token_stream)

    print("Building positional index...")
    positional_index = create_positional_index(token_stream)

    # Phase 2: Calculate statistical properties
    print("Calculating statistical properties...")
    df_stats = calculate_statistics(token_stream)
    print(df_stats)

    # Output example queries
    print(f"Documents containing both terms '{term1}' and '{term2}': {docs_and}")
    print(f"Documents containing either term '{term1}' or '{term2}': {docs_or}")
    print(f"Documents with terms '{term1}' and '{term2}' within {k} tokens: {docs_near}")
    print(f"Concordance for term 'climate':")
    for line in concordance_result:
        print(line)

if __name__ == "__main__":
    main()

# sujithkumaravel@MacBookAir Project1 % python3 new.py
# [nltk_data] Downloading package punkt to
# [nltk_data]     /Users/sujithkumaravel/nltk_data...
# [nltk_data]   Package punkt is already up-to-date!
# [nltk_data] Downloading package stopwords to
# [nltk_data]     /Users/sujithkumaravel/nltk_data...
# [nltk_data]   Package stopwords is already up-to-date!
# Extracting text from Reuters21578 dataset...
# Tokenizing text...
# Building inverted index...
# Building positional index...
# Calculating statistical properties...


# Preprocessing Step  Num Terms   ∆% Terms  ...  ∆% Postings  T% Terms  T% Postings
# 0         No Numbers         59  99.855069  ...    99.994578      0.14         0.01
# 1       Case Folding         59  99.855069  ...    99.994578      0.29         0.01
# 2      30 Stop Words         24  99.941045  ...    99.998152      0.35         0.01
# 3     150 Stop Words         59  99.855069  ...    99.994578      0.49         0.02
# 4           Stemming         50  99.877177  ...    99.994578      0.62         0.02
#
# [5 rows x 7 columns]



# Documents containing both terms 'bush' and 'reagan': {16672, 14146, 16643, 7, 7751, 8682, 9387, 16526, 4754, 3097, 14042}


# Documents containing either term 'bush' or 'reagan': {5, 7, 12301, 19, 25, 18460, 8233, 12333, 4143, 16439, 16444, 16446, 10303, 12353, 6221, 12366, 2138, 16474, 6236, 14434, 18530, 6255, 2160, 2163, 6265, 2175, 12418, 18571, 16526, 4241, 10389, 18586, 16540, 18592, 14497, 4260, 2218, 8366, 16559, 18608, 16564, 14517, 2230, 6327, 2239, 16575, 6340, 16582, 6345, 14537, 16589, 14542, 2257, 16599, 18650, 2268, 4316, 8417, 8418, 8421, 14566, 8424, 238, 16622, 8434, 8435, 18674, 6389, 12540, 18685, 258, 16643, 2307, 16647, 2317, 10510, 10511, 10513, 14609, 18710, 6428, 16671, 16672, 6433, 2338, 18724, 6438, 296, 10539, 4399, 2358, 18750, 16706, 337, 16745, 18795, 10612, 4470, 4475, 16764, 6525, 10621, 2432, 4487, 2441, 10635, 6546, 16787, 16789, 8603, 2465, 16802, 2467, 12709, 12715, 12716, 14767, 435, 2486, 8631, 8636, 16830, 449, 16833, 16835, 2501, 16838, 8647, 18890, 8654, 465, 18905, 6621, 10720, 482, 6631, 8682, 6635, 500, 6646, 4599, 12793, 14842, 515, 14855, 8722, 8724, 18964, 12831, 18977, 12834, 18982, 8743, 554, 18995, 8757, 16950, 566, 567, 6712, 8758, 16953, 16954, 574, 6721, 12865, 10821, 2639, 19028, 4699, 605, 4703, 17001, 2669, 8813, 8817, 2678, 17018, 2688, 2690, 4752, 4753, 4754, 658, 659, 4755, 4756, 2711, 12950, 6809, 4762, 4763, 8858, 12954, 17052, 8866, 10923, 15020, 15021, 10926, 12975, 689, 8883, 8884, 8887, 6840, 10949, 10959, 17105, 2772, 13012, 13013, 13014, 4827, 10972, 4829, 4830, 4831, 6877, 15081, 4845, 13037, 4849, 13049, 2810, 11008, 13058, 13059, 782, 8974, 15120, 17172, 13077, 800, 13096, 11052, 2866, 6962, 2869, 15158, 6969, 17211, 15165, 6974, 2884, 17226, 843, 6992, 6999, 9056, 11105, 4963, 13171, 11125, 11128, 7035, 900, 5002, 17296, 7062, 17309, 17318, 9129, 9134, 2992, 2994, 9138, 949, 15287, 7100, 15304, 15311, 17370, 15323, 13278, 5090, 17380, 13291, 1008, 9202, 15350, 9209, 15353, 3070, 15362, 11276, 15377, 15378, 9236, 11286, 9239, 3097, 3100, 7196, 15395, 9254, 7209, 15401, 9259, 11308, 3117, 15406, 17449, 15409, 7223, 1080, 17466, 1099, 15452, 13412, 1125, 13414, 17511, 15466, 1131, 1132, 1138, 3190, 5242, 3197, 15491, 1166, 9362, 11410, 13463, 5282, 9378, 11427, 11431, 3242, 9387, 9389, 1210, 1213, 11456, 11458, 9415, 11469, 5329, 11476, 3287, 11489, 11491, 17636, 5353, 5364, 1270, 11519, 11525, 15630, 11535, 15631, 1301, 17688, 17690, 15647, 3360, 1320, 11560, 5421, 13614, 11572, 11573, 11574, 5431, 17723, 1342, 11585, 5445, 17738, 1371, 5475, 5485, 7534, 3442, 7542, 7546, 7548, 7551, 5536, 9641, 1458, 7607, 3514, 9658, 15804, 9679, 1493, 1498, 3546, 9693, 9695, 5600, 1509, 1518, 5622, 11775, 15874, 11783, 7705, 3615, 3616, 11811, 11815, 17964, 9783, 1595, 1599, 1600, 7745, 7751, 5703, 5704, 7753, 5707, 17996, 17997, 7758, 15950, 11859, 7765, 5720, 1625, 1626, 3679, 5728, 9823, 15970, 15971, 3686, 7784, 9833, 15977, 15981, 7793, 5758, 5763, 13965, 7825, 9880, 7833, 18076, 7840, 13985, 13991, 11944, 13994, 1714, 7864, 18106, 7872, 7875, 5828, 18126, 18131, 7892, 3797, 7894, 7897, 14042, 14044, 18142, 7903, 18145, 5858, 14057, 9962, 18160, 14065, 18164, 7925, 16118, 1783, 14079, 7938, 18181, 9990, 1803, 5899, 9997, 7950, 10007, 10011, 3874, 3875, 18215, 5933, 10031, 5936, 10040, 3898, 18238, 16192, 14145, 14146, 5954, 10051, 16198, 10056, 12105, 10064, 10065, 1877, 10069, 10072, 5977, 10074, 3931, 3932, 14168, 18277, 18281, 10091, 10093, 10099, 10100, 6015, 18303, 16259, 10120, 6025, 18314, 10130, 12178, 12180, 18327, 18329, 16283, 1948, 18332, 6048, 10145, 8100, 1959, 10151, 1961, 10153, 10158, 1967, 10162, 10163, 14268, 4029, 1990, 14281, 6093, 10191, 2007, 10200, 10203, 18395, 18398, 2017, 6114, 18401, 10215, 14311, 18410, 4075, 4079, 12273, 10226, 12278, 10231, 6136, 14326, 4090, 12283, 18422, 2045}


# Documents with terms 'bush' and 'reagan' within 5 tokens: [7751, 8682, 14042, 14146]


# Concordance for term 'climate':
# 96: meat livestock corp peter frawley said told reuters improvement economic climate less competition european community lead gulf area higher beef sales
# 282: recent years community new york state working hard improve business climate manufucturing said william fowble kodak sernior general manager manufacturing assume
# 843: control law failure said would jeopardize continuation current favorable economic climate reuter
# 1124: rapidly expanding money supply low dollar development could worsen business climate increasing uncertainty pushing interest rates turn would adversely affect world
# 1177: overlooked many offshore investors heavily buying australian gold resource stocks climate could prove ideal time float csr petroleum division said reuter
# 1330: rely support independent deputies four opposition parties united vote today climate economic austerity politicians would risk returning polls soon haughey could
# 1422: one account filling customer orders arisen many traders believe create climate trading abuses especially extremely volatile stock index futures pit adoption
# 1456: said station still functioning communist party leader harilaos florakis said climate calmer today greek newspapers reported greek army navy air force
# 2007: japanese money spare companies unwilling invest real plant machinery present climate stagnant growth would keep snapping stocks jump back winners jump
# 2041: potential created still natural human resources reward holding present economic climate could huge added reuter
# 2221: last year instead pct reported bangemann said could deny economic climate west germany cooled stressed country downtrend minister also criticised state
# 2235: details also said ireland wished provide right conditions favourable taxation climate developing international fnnancial services sector reuter
# 2566: precious metals climate improving says montagu climate precious metals improving prices benefiting renewed
# 2566: precious metals climate improving says montagu climate precious metals improving prices benefiting renewed inflation fears switching funds
# 2579: precious metals climate improving says montagu climate precious metals improving prices benefiting renewed
# 2579: precious metals climate improving says montagu climate precious metals improving prices benefiting renewed inflation fears switching funds
# 2872: street certificates adding panel supported government efforts create favourable investment climate reuter
# 3394: fiscal ended september said could predict much country uncertain financial climate brennan said expects bank growth come increased exports canada banco
# 3889: raymond stone ward mccarthy said fundamentals generally bode healthy investment climate market confidence environment illumination confidence policy salomon brothers henry kaufman
# 4311: stability protracted period difficulty banking commissioner robert fell said banking climate dramatically changed year ago said news conference present annual report
# 4558: stable dollar rate expected strong domestic demand led believe investment climate would remain friendly economy would continue slow sure growth bundesbank
# 4658: strategy reserves become available attractive prices company said expects marketing climate natural gas improve provide opportunity amoco expand sales prices demand
# 4764: basic reading writing mathematics skills urged help students adapt economic climate volcker said challenge greatest education minority groups blacks hispanics reuter
# 4852: basic reading writing mathematics skills urged help students adapt economic climate volcker said challenge greatest education minority groups blacks hispanics reuter
# 5080: terminated effort acquire corning natural gas cited reasons uncertain regulatory climate new york state depresed price new york state electric stock
# 6807: general outlook swiss economy remained favourable despite difficult international economic climate facing export industry centre repeated previous forecast growth swiss domestic
# 6842: bankers abruptly cut credit lines country concern deteriorating political economic climate pretoria last week announced reached rescheduling agreement major international creditors
# 6937: member norway membership rejected referendum government officials wondering whether political climate changed nowhere debate existential switzerland resisted alliances years swiss microcosm
# 7118: economic reality demands measures beyond sees politically practicable said political climate meant would continued monetary policy hold exchange rate maintain confidence
# 7168: pct compared pct diw ifo predicted three pct increase saw climate equipment investment improving predicted rise four pct pct ifo diw
# 7273: germany fears may seem exaggerated said often turned past price climate quickly deteriorate forcing central bank restrictive policy said economic costs
# 8903: trade policy continued attention currency management fairly low interest rate climate major tax increases essential ingredients outlook red cavaney american paper
# 8904: action necessary said number favourable indicators high level investment good climate consumption meant recovery could expected exports would pick slightly course
# 9590: ppg industries inc continue show earnings growth despite difficult economic climate company said annual report mailed shareholders today company earnings rose
# 9932: showed countries taken steps increase foreign aid number improved investment climate countries said made structural adjustments economies taken measures promote exports
# 9995: middle east bbme standard chartered also found last year difficult climate make strong profits bbme net profit fell mln riyals mln
# 10679: second best reformers drew ambitious plans impractical current political economic climate said chinese source said month ago premier zhao ziyang decided
# 10816: includes deferred income taxes respectively year net includes operations friedrich climate master inc company acquired august full period comparable period includes
# 12654: mergers individual firms would carve specialised niches ensure survival deregulated climate ogren said sector losses bad loans bound rise steeply due
# 13093: said said imports rose billion baht billion thailand improved business climate year resulted pct increase imports raw materials products country oil
# 13152: optimistic proposed mine pamplona negros oriental province well country investment climate sulphur project involve initial capital expenditures mln dlrs depending upon
# 13415: announcing wage price freeze argentine officials said country needed serene climate carry structural changes economy argentina suspend interest payments foreign debts
# 14043: declined pct month contracts exchange said relatively steady interest rate climate reduced volume active contract treasury bond futures pct year ago
# 14369: utility debt fecsa general manager jose zaforteza told reuters thought climate talks would greatly improved new tariff proposals go far wrong
# 14444: required produce additional output would around one billion dlrs political climate may make difficult raise money said soviet union whose exports
# 14925: survey said study grew concern mergers acquisitions seriously hurt economic climate northern california however hall said jobs lost region new jobs
# 15413: car air conditioners japan matsushita said statement new venture japan climate systems corp capitalised billion yen three partners hold equal shares
# 15463: contracts per day throughout surpassed contracts daily past two months climate could better venture cme said merrill lynch commodity marketing vice
# 16043: told oil industry tuesday extend contracts producing blocks improve investment climate wants see increased expenditure exploration return president suharto opening speech
# 16043: contracts present laws suharto said apart government keep improving investment climate order accelerate development petroleum industry said indonesian energy minister subroto
# 16514: developed countries debt program going happen difficult create security current climate gerstner said securitized products could structured either retail institutional markets
# 16872: manufacturers expect downturn survey majority australian manufacturers expect deterioration business climate according march survey industrial trends westpac banking corp confederation australian
# 16890: windfall profits taxes bass strait crude longer appropriate current economic climate story said maximum pct levy old oil discovered september forcing
# 16937: german business climate worsens slightly institute business climate west german manufacturing industry worsened
# 16937: german business climate worsens slightly institute business climate west german manufacturing industry worsened slightly last month deteriorating markedly
# 16937: months causing cut back production plans ifo blamed worsening business climate outlook largely expectations declines foreign demand quarter firms polled said
# 16937: quarter firms polled said thought orders hand insufficient overall business climate february could described short satisfactory ifo said breakdown responses industrial
# 16937: responses industrial sector institute said manufacturers basic products judged business climate slightly less favourable february january coming months expect export prospects
# 16937: reported slight improvement demand increasingly said orders hand insufficient business climate capital goods industry unchanged february compared january ifo said due
# 16937: generally described weak outlook february cloudier january institute said business climate consumer durables sector remained industry average february carmakers positive current
# 17340: deferred oil exploration development outlays helping santos deal adverse business climate santos said remained financially strong injection mln dlrs second instalment
# 17470: far east executives ordered investigation said fernandez action fitted political climate former president ferdinand marcos final years prevailing mood day high
# 17652: attention earnings results desire pursue longer term growth opportunities investment climate improve belief low volatile crude oil prices could continue next
# 18367: wartenberg said consumer demand remained quite good noted cooling investment climate certainly reason heightened watchfulness stimulative steps said best way bonn
