package recognizer;

import jmrc.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LeastMedSq;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.FileInputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Properties;
import java.util.Set;


/**
 * Created by pegah on 5/30/17.
 */
public class PersonlityModels {

    public static String classificationMode = "classification";//regression
    private static String factors = "Extraversion, Emotional stability, Agreeableness, Conscientiousness, Openness to experience";
    private static String featureList = "WC, WPS, UNIQUE, SIXLTR, ABBREVIATIONS, EMOTICONS, QMARKS, PERIOD, COMMA, COLON, SEMIC, QMARK, EXCLAM, DASH, QUOTE, APOSTRO, PARENTH, OTHERP, ALLPCT, PRONOUN, I, WE, SELF, YOU, OTHER, NEGATE, ASSENT, ARTICLE, PREPS, NUMBER, AFFECT, POSEMO, POSFEEL, OPTIM, NEGEMO, ANX, ANGER, SAD, COGMECH, CAUSE, INSIGHT, DISCREP, INHIB, TENTAT, CERTAIN, SENSES, SEE, HEAR, FEEL, SOCIAL, COMM, OTHREF, FRIENDS, FAMILY, HUMANS, TIME, PAST, PRESENT, FUTURE, SPACE, UP, DOWN, INCL, EXCL, MOTION, OCCUP, SCHOOL, JOB, ACHIEVE, LEISURE, HOME, SPORTS, TV, MUSIC, MONEY, METAPH, RELIG, DEATH, PHYSCAL, BODY, SEXUAL, EATING, SLEEP, GROOM, SWEAR, NONFL, FILLERS, DIC, NLET, NPHON, NSYL, K_F_FREQ, K_F_NCATS, K_F_NSAMP, T_L_FREQ, BROWN_FREQ, FAM, CONC, IMAG, MEANC, MEANP, AOA";


    public static void main(String[] args) {
        if (classificationMode.equals("classification")) {
            trainClassificationModel(new File("examples"));
        } else {
            trainRegressionModel(new File("examples"));
        }

    }

    private static void trainRegressionModel(File dir) {

        try {
            String[] featureAttributes = featureList.split(",");
            FastVector fv = new FastVector(featureAttributes.length + 5);

            // create feature vector for training------------------
            //Class
            FastVector fv_class = new FastVector(5);
            for (String factor : factors.split(","))
                fv.addElement(new Attribute(factor));


            //Attributes
            for (String attr : featureAttributes) {
                fv.addElement(new Attribute(attr));
            }

            //fill train data------------------------------------
            Instances trainData = new Instances("Rel", fv, dir.listFiles().length);
            for (File file : dir.listFiles()) {
                //for each file we have an instance
                int index = 0;
                String text = Utils.readFile(file);
                Map<String, Double> features = getFeatureExtractionFromText(text);
                Instance participantInstanceData = new Instance(features.keySet().size() + 5);

                //class
                Map<String, Double> tags = getRegressionTagExtractionFromText(text);
                for (Map.Entry<String, Double> entry : tags.entrySet()) {
                    participantInstanceData.setValue((Attribute) fv.elementAt(index), entry.getValue());
                    index++;
                }

                //attributes
                for (Map.Entry<String, Double> entry : features.entrySet()) {
                    participantInstanceData.setValue((Attribute) fv.elementAt(index), entry.getValue());
                    index++;
                }
                trainData.add(participantInstanceData);
            }

            //train model--------------------------------
            Classifier cls = new LinearRegression();
            cls.buildClassifier(trainData);

            //test model---------------------------------
            Instances isTestingSet = null;
            Evaluation eTest = new Evaluation(isTestingSet);
            eTest.evaluateModel(cls, isTestingSet);
            String strSummary = eTest.toSummaryString();
            System.out.println(strSummary);

        } catch (Exception exception) {
            System.out.print(exception.getMessage());
        }
    }

    private static void trainClassificationModel(File dir) {

        try {
            String[] featureAttributes = featureList.split(",");
            FastVector fv = new FastVector(featureAttributes.length + 1);

            // create feature vector for training------------------
            //Class
            FastVector fv_class = new FastVector(5);
            for (String factor : factors.split(","))
                fv_class.addElement(factor);
            Attribute ClassAttribute = new Attribute("theClass", fv_class);
            fv.addElement(ClassAttribute);

            //Attributes
            for (String attr : featureAttributes) {
                fv.addElement(new Attribute(attr));
            }

            //fill train data------------------------------------
            Instances trainData = new Instances("Rel", fv, dir.listFiles().length);
            trainData.setClassIndex(0);
            for (File file : dir.listFiles()) {
                //for each file we have an instance
                int index = 0;
                String text = Utils.readFile(file);
                Map<String, Double> features = getFeatureExtractionFromText(text);
                Instance participantInstanceData = new Instance(features.keySet().size() + 1);

                //class
                String tag = getClassificationTagExtractionFromText(text);
                participantInstanceData.setValue((Attribute) fv.elementAt(index), tag);
                index++;

                //attributes
                for (Map.Entry<String, Double> entry : features.entrySet()) {
                    participantInstanceData.setValue((Attribute) fv.elementAt(index), entry.getValue());
                    index++;
                }
                trainData.add(participantInstanceData);
            }

            //train model--------------------------------
            Classifier cls = new J48();
            cls.buildClassifier(trainData);

            //test model---------------------------------
            Instances isTestingSet = null;
            Evaluation eTest = new Evaluation(isTestingSet);
            eTest.evaluateModel(cls, isTestingSet);
            String strSummary = eTest.toSummaryString();
            System.out.println(strSummary);

        } catch (Exception exception) {
            System.out.print(exception.getMessage());
        }
    }

    private static Map<String, String> getShortFeatureNames() {

        Map<String, String> shortcuts = new LinkedHashMap<String, String>();

        shortcuts.put("LINGUISTIC", "LINGUISTIC");
        shortcuts.put("PRONOUN", "PRONOUN");
        shortcuts.put("I", "I");
        shortcuts.put("WE", "WE");
        shortcuts.put("SELF", "SELF");
        shortcuts.put("YOU", "YOU");
        shortcuts.put("OTHER", "OTHER");
        shortcuts.put("NEGATIONS", "NEGATE");
        shortcuts.put("ASSENTS", "ASSENT");
        shortcuts.put("ARTICLES", "ARTICLE");
        shortcuts.put("PREPOSITIONS", "PREPS");
        shortcuts.put("NUMBERS", "NUMBER");
        shortcuts.put("PSYCHOLOGICAL PROCESS", "PSYCHOLOGICAL PROCESS");
        shortcuts.put("AFFECTIVE PROCESS", "AFFECT");
        shortcuts.put("POSITIVE EMOTION", "POSEMO");
        shortcuts.put("POSITIVE FEELING", "POSFEEL");
        shortcuts.put("OPTIMISM", "OPTIM");
        shortcuts.put("NEGATIVE EMOTION", "NEGEMO");
        shortcuts.put("ANXIETY", "ANX");
        shortcuts.put("ANGER", "ANGER");
        shortcuts.put("SADNESS", "SAD");
        shortcuts.put("COGNITIVE PROCESS", "COGMECH");
        shortcuts.put("CAUSATION", "CAUSE");
        shortcuts.put("INSIGHT", "INSIGHT");
        shortcuts.put("DISCREPANCY", "DISCREP");
        shortcuts.put("INHIBITION", "INHIB");
        shortcuts.put("TENTATIVE", "TENTAT");
        shortcuts.put("CERTAINTY", "CERTAIN");
        shortcuts.put("SENSORY PROCESS", "SENSES");
        shortcuts.put("SEEING", "SEE");
        shortcuts.put("HEARING", "HEAR");
        shortcuts.put("FEELING", "FEEL");
        shortcuts.put("SOCIAL PROCESS", "SOCIAL");
        shortcuts.put("COMMUNICATION", "COMM");
        shortcuts.put("REFERENCE PEOPLE", "OTHREF");
        shortcuts.put("FRIENDS", "FRIENDS");
        shortcuts.put("FAMILY", "FAMILY");
        shortcuts.put("HUMANS", "HUMANS");
        shortcuts.put("RELATIVITY", "RELATIVITY");
        shortcuts.put("TIME", "TIME");
        shortcuts.put("PAST", "PAST");
        shortcuts.put("PRESENT", "PRESENT");
        shortcuts.put("FUTURE", "FUTURE");
        shortcuts.put("SPACE", "SPACE");
        shortcuts.put("UP", "UP");
        shortcuts.put("DOWN", "DOWN");
        shortcuts.put("INCLUSIVE", "INCL");
        shortcuts.put("EXCLUSIVE", "EXCL");
        shortcuts.put("MOTION", "MOTION");
        shortcuts.put("PERSONAL PROCESS", "PERSONAL PROCESS");
        shortcuts.put("OCCUPATION", "OCCUP");
        shortcuts.put("SCHOOL", "SCHOOL");
        shortcuts.put("JOB OR WORK", "JOB");
        shortcuts.put("ACHIEVEMENT", "ACHIEVE");
        shortcuts.put("LEISURE ACTIVITY", "LEISURE");
        shortcuts.put("HOME", "HOME");
        shortcuts.put("SPORTS", "SPORTS");
        shortcuts.put("TV OR MOVIE", "TV");
        shortcuts.put("MUSIC", "MUSIC");
        shortcuts.put("MONEY", "MONEY");
        shortcuts.put("METAPHYSICAL", "METAPH");
        shortcuts.put("RELIGION", "RELIG");
        shortcuts.put("DEATH AND DYING", "DEATH");
        shortcuts.put("PHYSICAL STATES", "PHYSCAL");
        shortcuts.put("BODY STATES", "BODY");
        shortcuts.put("SEXUALITY", "SEXUAL");
        shortcuts.put("EATING", "EATING");
        shortcuts.put("SLEEPING", "SLEEP");
        shortcuts.put("GROOMING", "GROOM");
        shortcuts.put("EXPERIMENTAL DIMENSION", "EXPERIMENTAL DIMENSION");
        shortcuts.put("SWEAR WORDS", "SWEAR");
        shortcuts.put("NONFLUENCIES", "NONFL");
        shortcuts.put("FILLERS", "FILLERS");

        return shortcuts;
    }

    private static String getClassificationTagExtractionFromText(String text) throws Exception {

        int start = text.indexOf('#') + 1;
        int end = text.indexOf('#', start);
        String tags = text.substring(start, end);
        Map<String, String> counts = new LinkedHashMap<String, String>();
        String[] personalityClasses = tags.split(",");
        for (String factor : personalityClasses) {
            if (factor.split(":")[1].equals("1")) {
                return factor.split(":")[0];
            }
        }
        return null;
    }

    private static Map<String, Double> getRegressionTagExtractionFromText(String text) throws Exception {

        int start = text.indexOf('#') + 1;
        int end = text.indexOf('#', start);
        String tags = text.substring(start, end);
        Map<String, Double> counts = new LinkedHashMap<String, Double>();
        String[] personalityClasses = tags.split(",");
        for (String factor : personalityClasses) {
            counts.put(factor.split(":")[0], Double.parseDouble(factor.split(":")[1]));
        }
        return counts;
    }

    private static Map<String, Double> getFeatureExtractionFromText(String text) throws Exception {

        Properties properties = new Properties();
        properties.load(new FileInputStream("PersonalityRecognizer.properties"));
        File liwcCatFile = new File(properties.getProperty("liwcCatFile"));
        File mrcPath = new File(properties.getProperty("mrcPath"));

        Map<String, Double> counts = new LinkedHashMap<String, Double>();
        LIWCDictionary liwcDic = new LIWCDictionary(liwcCatFile);
        MRCDatabase mrcDb = new MRCDatabase(mrcPath);
        Map<String, String> featureShortcuts = getShortFeatureNames();

        //compute tags
        int start = text.indexOf('#') + 1;
        int end = text.indexOf('#', start);
        String tags = text.substring(start, end);
        text = text.replace("#" + tags + "#", "");

        // compute LIWC and MRC features
        Map<String, Double> initCounts = liwcDic.getCounts(text, true);

        for (String longFeature : initCounts.keySet()) {

            if (featureShortcuts.containsKey(longFeature)) {
                counts.put(featureShortcuts.get(longFeature), initCounts.get(longFeature));
            } else {
                counts.put(longFeature, initCounts.get(longFeature));
                // System.err.println("Warning: LIWC feature " + longFeature + " not recognized, check LIWC.CAT file");
            }
        }

        // compute MRC features
        Map<String, Double> mrcFeatures = getMRCCounts(mrcDb, text);
        counts.putAll(mrcFeatures);
        System.err.println("MRC features computed: " + mrcFeatures.size());

        return counts;
    }

    private static Map<String, Double> getMRCCounts(MRCDatabase db, String text) throws QueryException {

        Field[] MRC_FEATURES = {MRCDatabase.FIELD_NLET, MRCDatabase.FIELD_NPHON,
                MRCDatabase.FIELD_NSYL, MRCDatabase.FIELD_K_F_FREQ, MRCDatabase.FIELD_K_F_NCATS,
                MRCDatabase.FIELD_K_F_NSAMP, MRCDatabase.FIELD_T_L_FREQ, MRCDatabase.FIELD_BROWN_FREQ,
                MRCDatabase.FIELD_FAM, MRCDatabase.FIELD_CONC, MRCDatabase.FIELD_IMAG,
                MRCDatabase.FIELD_MEANC, MRCDatabase.FIELD_MEANP, MRCDatabase.FIELD_AOA};

        MRCPoS[] MRC_POS = {
                MRCPoS.NOUN, MRCPoS.VERB, MRCPoS.ADJECTIVE, MRCPoS.ADVERB, MRCPoS.PAST_PARTICIPLE,
                MRCPoS.PREPOSITION, MRCPoS.CONJUNCTION, MRCPoS.PRONOUN, MRCPoS.INTERJECTION, MRCPoS.OTHER};

        // tokenize text
        String[] words = LIWCDictionary.tokenize(text);

        Map<String, Double> counts = new LinkedHashMap<String, Double>(MRC_FEATURES.length);
        Map<String, Integer> nonzeroWords = new LinkedHashMap<String, Integer>(MRC_FEATURES.length);

        // initialize counts
        for (int i = 0; i < MRC_FEATURES.length; i++) {
            counts.put(MRC_FEATURES[i].toString(), 0.0);
            nonzeroWords.put(MRC_FEATURES[i].toString(), 0);
        }

        for (int i = 0; i < words.length; i++) {

            if (db.containsWord(words[i])) {
                Set<PoS> posSet = db.getAvailablePoS(words[i]);

                // only consider the first PoS in the order specified by MRC_POS
                for (int j = 0; j < MRC_POS.length; j++) {

                    if (posSet.contains(MRC_POS[j])) {
                        // update counts for all fields in the database
                        for (int k = 0; k < MRC_FEATURES.length; k++) {
                            try {
                                counts.put(MRC_FEATURES[k].toString(), counts
                                        .get(MRC_FEATURES[k].toString())
                                        + db.getValue(words[i], MRC_POS[j],
                                        MRC_FEATURES[k]));
                                // update non zero words if no exception
                                nonzeroWords.put(MRC_FEATURES[k].toString(),
                                        nonzeroWords.get(MRC_FEATURES[k]
                                                .toString()) + 1);

                            } catch (UndefinedValueException e) {
                                // proceed to next field
                            } catch (EntryNotFoundException e) {
                                System.err.println("Warning: entry " + words[i]
                                        + "/" + MRC_POS[j].toString() + "/"
                                        + MRC_FEATURES[k].toString()
                                        + " not found");
                            }
                        }
                        // PoS matched, proceed to next word
                        break;
                    }
                }
            }
        }

        // get ratio of feature counts over all non zero words
        for (String feature : counts.keySet()) {
            if (nonzeroWords.get(feature) != 0) {
                counts.put(feature, counts.get(feature)
                        / nonzeroWords.get(feature));
            } else {
                counts.put(feature, Double.NaN);
            }
        }

        return counts;
    }


}
