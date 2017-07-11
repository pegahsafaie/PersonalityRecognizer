package recognizer;

import java.io.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import jmrc.EntryNotFoundException;
import jmrc.Field;
import jmrc.MRCDatabase;
import jmrc.MRCPoS;
import jmrc.PoS;
import jmrc.QueryException;
import jmrc.UndefinedValueException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.core.*;

/**
 * <pre>
 * 
 *  
 *  The program computes features described in (<a href=http://www.mairesse.co.uk/papers/personality-jair07.pdf target=_top>Mairesse et al., 2007</a>) 
 *  given a text, and it runs Weka models on the features to produce 
 *  personality scores for all Big Five dimensions.
 *  
 *  The MRC Psycholinguistic database and the LIWC tool need to be installed, 
 *  and the file PersonalityRecognizer.conf in the main directory needs to 
 *  be modified accordingly. The PersonalityRecognizer script should be used 
 *  for launching the program.
 * 
 *  Usage: PersonalityRecognizer [-d] [-m model_number] [-o] [-c] [-t model_type] [-a arff_output_file] -i file|directory
 *		-c,--counts      Also outputs feature counts, -d must be disabled
 *		-d,--directory   Corpus analysis mode. Input must be a directory with 
 * 				 multiple text files, features are standardized over 
 * 				 the corpus and the recognizer outputs a personality 
 * 				 estimate for each text file.
 * 		-i,--input       Input file or directory (required)
 *		-m,--model       Model to use for computing scores (default 4). Options:
 *	              				1 = Linear Regression
 *  	           				2 = M5' Model Tree
 *             					3 = M5' Regression Tree
 *             					4 = Support Vector Machine with Linear Kernel (SMOreg)
 * 		-o,--outputmod   Also outputs models
 * 		-t,--type	 Selects the type of model to use (default 1). The appropriate
 * 				 model depends on the language sample (written or 
 * 				 spoken), and whether observed personality (as perceived 
 *  				 by external judges) or self-assessed personality (the 
 * 				 writer/speaker's perception) needs to be estimated from the 
 * 				 text. Options:
 * 						1 = Observed personality from spoken language
 * 						2 = Self-assessed personality from written language
 *     		-a,--arff	In corpus analysis mode, outputs the features of each text into 
 * 				a Weka <code>.arff</code> dataset file, together with the predicted scores.
 * 				New models can be trained by adding features and replacing the scores
 * 				with human estimates. Each line corresponds to a text in the corpus 
 * 				indicated by the <code>filename</code> feature.
 *   
 *  
 *  See the included readme file and the website 
 *  <a href=http://www.mairesse.co.uk/personality/recognizer.html target=_top>http://www.mairesse.co.uk/personality/recognizer.html</a>
 *  for more information. 
 *  
 *  Questions can be emailed to the author (webpage: http://www.mairesse.co.uk).
 *  
 *  Reference paper:
 *  
 *  Francois Mairesse, Marilyn Walker, Matthias Mehl and Roger Moore. 
 *  Using Linguistic Cues for the Automatic Recognition of Personality in 
 *  Conversation and Text. Journal of Artificial Intelligence Research (JAIR), 
 *  30, pages 457-500, 2007. 
 *  
 *  Available on the web in PDF format at 
 *  <a href=http://www.mairesse.co.uk/papers/personality-jair07.pdf target=_top>http://www.mairesse.co.uk/papers/personality-jair07.pdf</a>
 *  
 * </pre>
 * 
 * @author Francois Mairesse, <a href=http://www.mairesse.co.uk/
 *         target=_top>http://www.mairesse.co.uk</a>
 * @version 1.03
 *  
 */

public class PersonalityRecognizer {

	//----------------------------
	// Global variables to be specified in the PersonalityRecognizer.conf file
	// and initalized in the initializeParameter() method

	/** Main application directory (in configuration file). */
	private File appDir;

	/**
	 * LIWC.CAT dictionary file from the Linguistic Inquiry and Word Count
	 * (LIWC) tool (in configuration file).
	 */
	private File liwcCatFile;

	/**
	 * Path to the mrc2.dct file from the MRC Psycholinguistic Database (in configuration
	 * file).
	 */
	private File mrcPath;

	/**
	 * Weka ARFF file with all attributes and no instance 
	 */
	private File attributeFile;

	//----------------------------

	/** Available Weka model names. */
	private static final String[] MODEL_NAMES = { "Linear Regression", "M5' Model Tree",
			"M5' Regression Tree", "Support Vector Machine with Linear Kernel (SMOreg)" };

	/**
	 * Weka model directories (under the lib/models subdirectory) corresponding
	 * to each element of the MODEL_NAMES array.
	 */
	private static final String[] MODEL_DIRS = { "LinearRegression", "M5P", "M5P-R",
			"SVM" };

	/**
	 * Default model index in the MODEL_NAMES array (default is 1 = M5' Model
	 * Tree).
	 */
	private int DEFAULT_MODEL = 3;

	/** Personality dimensions names. */
	public static final String[] DIMENSIONS = { "Extraversion",
			"Emotional stability", "Agreeableness", "Conscientiousness",
			"Openness to experience" };

	/**
	 * Weka personality model files for each trait in the DIMENSION array. All
	 * the files specified need to be present in each model's directory.
	 */
	private static final String[] DIM_MODEL_FILES = { "extra.model", "ems.model",
			"agree.model", "consc.model", "open.model" };

	/**
	 * MRC feature names in the model files and in the MRC Psycholinguistic
	 * Database.
	 */
	private static final Field[] MRC_FEATURES = { MRCDatabase.FIELD_NLET, MRCDatabase.FIELD_NPHON,
			MRCDatabase.FIELD_NSYL, MRCDatabase.FIELD_K_F_FREQ, MRCDatabase.FIELD_K_F_NCATS,
			MRCDatabase.FIELD_K_F_NSAMP, MRCDatabase.FIELD_T_L_FREQ, MRCDatabase.FIELD_BROWN_FREQ,
			MRCDatabase.FIELD_FAM, MRCDatabase.FIELD_CONC, MRCDatabase.FIELD_IMAG,
			MRCDatabase.FIELD_MEANC, MRCDatabase.FIELD_MEANP, MRCDatabase.FIELD_AOA };

	/** Valid PoS tags in the MRC Psycholinguistic Database. */
	private static final MRCPoS[] MRC_POS = { 
			MRCPoS.NOUN, MRCPoS.VERB, MRCPoS.ADJECTIVE, MRCPoS.ADVERB, MRCPoS.PAST_PARTICIPLE, 
			MRCPoS.PREPOSITION, MRCPoS.CONJUNCTION, MRCPoS.PRONOUN, MRCPoS.INTERJECTION, MRCPoS.OTHER };
		

	/** Line separator. */
	public static final String LS = System.getProperty("line.separator");

	/** File separator. */
	public static final String FS = File.separator;

	/**
	 * Configuration file (default is PersonalityRecognizer.conf in root
	 * application directory).
	 */
	public static final File DEFAULT_CONFIG_FILE = new File(
			"PersonalityRecognizer.properties");

	/** MRC Psycholinguistic database. */
	private MRCDatabase mrcDb;
	
	/** LIWC dictionary. */
	private LIWCDictionary liwcDic;
	
	/** Mapping between long feature names and short ones in the Weka models. */
	private Map<String,String> featureShortcuts;
	
	/** Arrays of features that aren't included in the models. **/
	private static final String[] domainDependentFeatures = {"FRIENDS", "FAMILY", 
		"OCCUP", "SCHOOL", "JOB", "LEISURE", "HOME","SPORTS","TV", "MUSIC", "MONEY", 
		"METAPH", "DEATH", "PHYSCAL", "BODY", "EATING", "SLEEP", "GROOM"};
	
	/** Set of features that aren't included in the models. **/
	private Set<String> domainDependentFeatureSet;
	
	/** Arrays of features that aren't included in one instance analysis (corpus analysis only). **/
	private static final String[] absoluteCountFeatures = {"WC"};
	
	/** Set of features that aren't included in one instance analysis (corpus analysis only). **/
	private Set<String> absoluteCountFeatureSet;
	
	
	/**
	 * 
	 * Main method that initializes the parameters from the configuration file,
	 * counts the features from the input text(s), run the specified Weka models
	 * for this feature set for each Big Five personality traits, and returns
	 * the personality score estimates to the standard output.
	 * 
	 * @param args
	 *            set of options and input file(s).
	 */
	public static void main(String[] args) {

		try {

			// get options
			Options options = new Options();
			options.addOption("i", "input", true,
					"Input file or directory (required)");
			options
					.addOption("d", "directory", false,
							"Corpus analysis mode. Input must be a directory with " + LS + 
							"multiple text files, features are standardized over " + LS  + 
							"the corpus and the recognizer outputs a personality " + LS  + 
							"estimate for each text file.");
			options.addOption("m", "model", true,
					"Model to use for computing scores (default 4). Options: "
							+ LS + "    1 = Linear Regression" + LS
							+ "    2 = M5' Model Tree" + LS
							+ "    3 = M5' Regression Tree" + LS
							+ "    4 = Support Vector Machine with Linear Kernel (SMOreg)");
			options.addOption("c", "counts", false,
					"Also outputs feature counts, -d must be disabled");
			options.addOption("o", "outputmod", false, "Also outputs models");
			options
					.addOption("t", "type", true,
							"Selects the type of model to use (default 1). " + LS
									+ "The appropriate model depends on the language sample" +LS + 
									"(written or spoken), and whether observed personality " + LS + 
									"(as perceived by external judges) or self-assessed " + LS + 
									"personality (the writer/speaker's perception) needs to " + LS + 
									"be estimated from the text. Options: "
									+ LS
									+ "    1 = Observed personality from spoken language"
									+ LS
									+ "    2 = Self-assessed personality from written language");
			
			options.addOption("a", "arff", true, 	"In corpus analysis mode, outputs the features of each" +LS + 
													"text into a Weka .arff dataset file," + LS + 
													"together with the predicted scores. New models can be" + LS  +
													"trained by adding features and replacing the scores" + LS + 
													"with human estimates. Each line corresponds to a text" + LS + "" +
													"in the corpus indicated by the filename" + LS  + "feature.");
			CommandLine cmd = null;
			CommandLineParser parser = new PosixParser();
			HelpFormatter help = new HelpFormatter();
			try {
				cmd = parser.parse(options, args);
			} catch (org.apache.commons.cli.ParseException e) {
				e.printStackTrace();
				help.printHelp("PersonalityRecognizer", options, true);
				System.exit(1);
			}
			if (!cmd.hasOption("i")) {
				help.printHelp("PersonalityRecognizer", options, true);
				System.exit(1);
			}

			// create and initialize recognizer using configuration file
			PersonalityRecognizer recognizer = new PersonalityRecognizer(
					DEFAULT_CONFIG_FILE);
			
			// set model if specified
			if (cmd.hasOption("m")) {
				recognizer
						.setModel(Integer.parseInt(cmd.getOptionValue("m")) - 1);
			}
			
			boolean selfModel = cmd.hasOption("t") && cmd.getOptionValue("t").equals("2");
			
			// load models for all Big Five traits
			Classifier[] models = recognizer.loadWekaModels(selfModel, cmd.hasOption("d"));
			
			// read input text from file or directory
			File inputFile = new File(cmd.getOptionValue("i"));
			
			if (!inputFile.exists()) {
				System.err.println("Error: input file or directory " + inputFile.getAbsolutePath() + " doesn't exist.");
				System.exit(1);
			}
			
			if (cmd.hasOption("d")) {
				
				// corpus analysis mode
				
				if (!inputFile.isDirectory()) {
					System.err.println("Error: -d switch is enabled but input file " + inputFile.getAbsolutePath() + " isn't a directory.");
					System.exit(1);
				}
				
				File outputArffFile = null;
				
				if (cmd.hasOption("a")) {
						
					outputArffFile = new File((cmd.getOptionValue("a")));
					if (!outputArffFile.getAbsoluteFile().getParentFile().exists()) {
						System.err.println("Error: can't write to arff file " + outputArffFile.getAbsolutePath());
						System.exit(1);
					}
						
				}
				// read directory to string
				System.err.println("Reading directory "
						+ inputFile.getAbsolutePath() + "...");
				
				// need to build dataset
				Map<File,Double[]> scores = recognizer.computeScoresOverCorpus(inputFile, models, outputArffFile);
			
				recognizer.printOutput(models, scores, recognizer
						.getModelIndex(), cmd.hasOption("o"), selfModel, System.out);
				
			} else {
				
				// single instance mode
				// requires non standardized models
				
				if (cmd.hasOption("a")) {
						System.err.println("Error: -d switch must be enabled for writing output arff file.");
						System.exit(1);						
				}
				
				if (inputFile.isDirectory()) {
					System.err.println("Error: -d switch is not enabled but input file " + inputFile.getAbsolutePath() + " is a directory.");
					System.exit(1);
				}
				
				
				// read single file to string
				String text = Utils.readFile(inputFile);
			
				// get feature counts from the input text
				Map<String,Double> counts = recognizer.getFeatureExtractionFromText(text, true);
				System.err.println("Total features computed: " + counts.size());
	
				// print feature counts
				if (cmd.hasOption("c")) {
					System.out.println("Feature counts:");
					Utils.printMap(counts, System.out);
				}
				
	
				// compute the personality scores of the new instance for each trait
				System.err.println("Running models...");
				double[] scores = recognizer.runWekaModels(models, counts);
	
				// print resulting scores
				recognizer.printOutput(models, scores, recognizer
							.getModelIndex(), cmd.hasOption("o"), selfModel, System.out);
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	/**
	 * Initializes parameters based on configuration file, and loads the LIWC
	 * dictionary and the MRC database in memory.
	 * 
	 * @param propFile
	 *            configuration file in ASCII format ( <i>VARIABLE = "VALUE"
	 *            </i> on each line).
	 */
	public PersonalityRecognizer(File propFile) {

		try {

			// initialize parameters
			loadProperties(propFile);

			// load LIWC dictionary in memory
			liwcDic = new LIWCDictionary(liwcCatFile);
			
			// load MRC database in memory
			mrcDb = new MRCDatabase(mrcPath) ;

			// load shortcut map
			featureShortcuts = getShortFeatureNames();
			
			// load non general features
			domainDependentFeatureSet = new LinkedHashSet<String>(Arrays.asList(domainDependentFeatures));

			// load absolute features
			absoluteCountFeatureSet = new LinkedHashSet<String>(Arrays.asList(absoluteCountFeatures));

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Initializes parameters based on the default configuration file
	 * (PersonalityRecognizer.properties).
	 */
	public PersonalityRecognizer() {
		this(DEFAULT_CONFIG_FILE);
	}
	
	
	/**
	 * Loads parameters from configuration file. This method is typically called
	 * from the constructor. Required parameters are (1) the application
	 * directory, (2) the path to the LIWC dictionary file LIWC.CAT, and (3) the mrc2.dct file  
	 * of the MRC Psycholinguistic Database.
	 * 
	 * @param propFile
	 *            Java property file in ASCII format ( <i>VARIABLE = VALUE
	 *            </i> on each line).
	 */
	private void loadProperties(File propFile) {
		// load properties
		Properties properties = new Properties();
		try {
			properties.load(new FileInputStream(propFile));
			appDir = new File(properties.getProperty("appDir"));
			liwcCatFile = new File(properties.getProperty("liwcCatFile"));
			mrcPath =  new File(properties.getProperty("mrcPath"));
			
			// set other global variables
			attributeFile = new File("/home/pegah/Documents/thesis/PersonalityRecognizer/lib/attributes-info.arff");

		} catch (Exception e) {
			System.err
					.println("Error: file not found, please edit the file "
							+ propFile.getName()
							+ " and make sure that the directory structure of the application is correct.");
			System.err.println("Exception message:");
			System.err.println(e.getMessage());
			System.exit(1);
		}
	}

	

	/**
	 * Sets the default Weka model to load when calling loadWekaModels().
	 * 
	 * @param modelIndex
	 *            the index of the element in the MODEL_DIRS array corresponding
	 *            to the directory of the model to load.
	 */
	public void setModel(int modelIndex) {
		if (modelIndex < MODEL_DIRS.length) {
			DEFAULT_MODEL = modelIndex;
		} else {
			System.err.println("Error: invalid model index (maximum + "
					+ (MODEL_DIRS.length - 1) + ")");
		}
	}

	/**
	 * Sets the default Weka model to load when calling loadWekaModels().
	 * 
	 * @param modelDir
	 *            the model subdirectory in the MODEL_DIRS array corresponding
	 *            to the model to load.
	 */
	public void setModel(String modelDir) {
		try {
			DEFAULT_MODEL = Arrays.asList(MODEL_DIRS).indexOf(modelDir);
		} catch (Exception e) {
			System.err
					.println("Error: specified model subdirectory not in model list.");
			e.printStackTrace();
		}
	}

	/**
	 * Gets the current default model index.
	 * 
	 * @return index of the default model in the MODEL_NAMES array
	 */
	public int getModelIndex() {
		return DEFAULT_MODEL;
	}

	/**
	 * Gets the model index in the MODEL_NAMES array from a string
	 * representation.
	 * 
	 * @param modelDir
	 *            the model subdirectory in the MODEL_DIRS array corresponding
	 *            to the model to load.
	 * @return the index of the model in the MODEL_NAMES array.
	 */
	public int getModelIndex(String modelDir) {
		try {
			return Arrays.asList(MODEL_DIRS).indexOf(modelDir);
		} catch (Exception e) {
			System.err
					.println("Error: specified model subdirectory not in model list.");
			e.printStackTrace();
			return -1;
		}
	}

	/**
	 * Loads saved Weka models in memory for all personality dimensions, using
	 * the default model type.
	 * 
	 * @param selfModel if set to true, loads the self-report models.
	 * @param stdModels if set to true, loads the standardized models.
	 * @return an array of Weka models (Classifier objects) loaded from each
	 *         model filename in specified in the DIM_MODEL_FILES array.
	 */
	public Classifier[] loadWekaModels(boolean selfModel, boolean stdModels) {
		return loadWekaModels(DEFAULT_MODEL, selfModel, stdModels);
	}

	/**
	 * Loads saved Weka models in memory for all personality dimensions.
	 * 
	 * @param modelIndex
	 *            the index of the element in the MODEL_DIRS array corresponding
	 *            to the directory of the model to load.
	 * @param selfModel if set to true, loads the self-report models.
	 * @param stdModels if set to true, loads the standardized models.
	 * @return an array of Weka models (Classifier objects) loaded from each
	 *         model filename in specified in the DIM_MODEL_FILES array.
	 */
	public Classifier[] loadWekaModels(int modelIndex, boolean selfModel, boolean stdModels) {

		if (modelIndex >= MODEL_DIRS.length) {
			System.err.println("Error: invalid model index (maximum + "
					+ (MODEL_DIRS.length - 1) + ")");
			modelIndex = DEFAULT_MODEL;
		}
		
		// change file name for standardized models
		String stdPrefix = "";
		if (stdModels) { stdPrefix = "std-"; }
		String modelType = "obs";
		if (selfModel) { modelType = "self"; }
		
		Classifier[] models = new Classifier[DIM_MODEL_FILES.length];
		try {
			// for each personality trait
			for (int i = 0; i < DIM_MODEL_FILES.length; i++) {
				// get model class based on parameter string
				models[i] = loadWekaModel(new File(appDir.getAbsolutePath()
						+ FS + "lib" + FS + "models" + FS + modelType + FS 
						+ MODEL_DIRS[modelIndex] + FS + stdPrefix + DIM_MODEL_FILES[i]));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return models;
	}

	/**
	 * Runs each Weka model on a new instance created from the input feature
	 * counts, and outputs the resulting personality score.
	 * 
	 * @param models
	 *            array of Weka models (Classifier objects).
	 * @param counts
	 *            mapping of feature counts (Double objects), it must probide
	 *            a value for all attribute strings of the input models.
	 * @return an array containing a personality score for each model.
	 */
	public double[] runWekaModels(Classifier[] models, Map<String,Double> counts) {
		double[] scores = new double[models.length];

		try {
			// for each model
			for (int i = 0; i < models.length; i++) {
				// compute score based on loaded model and counts
				scores[i] = runWekaModel(models[i], counts);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return scores;
	}

	/**
	 * Computes the features from the input text (70 LIWC features and 14 from
	 * the MRC database).
	 * 
	 * @param text
	 *            input text.
	 * @param relativeOnly do not return absolute count features (WC), must be set to false if
	 *        standardized features are used (corpus analysis mode).
	 * @return mapping associating feature names strings with feature counts
	 *         (Double objects).
	 * @throws Exception
	 */
	public Map<String,Double> getFeatureExtractionFromText(String text, boolean relativeOnly) throws Exception {
		
		
		Map<String,Double> counts = new LinkedHashMap<String,Double>();

		// compute LIWC and MRC features
		Map<String,Double> initCounts = liwcDic.getCounts(text, true);
						
		for (String longFeature : initCounts.keySet()) { 
			
			if (featureShortcuts.containsKey(longFeature)) {
				counts.put(featureShortcuts.get(longFeature), initCounts.get(longFeature)); 
			} else {
				counts.put(longFeature, initCounts.get(longFeature)); 
				// System.err.println("Warning: LIWC feature " + longFeature + " not recognized, check LIWC.CAT file");
			}
		}
		
		// remove domain dependent LIWC features
		counts.keySet().removeAll(domainDependentFeatureSet);
		if (relativeOnly) { counts.keySet().removeAll(absoluteCountFeatureSet); }
		System.err.println("LIWC features computed: " + counts.size());
		
		// compute MRC features
		Map<String,Double> mrcFeatures = getMRCCounts(mrcDb, text);
		counts.putAll(mrcFeatures);
		System.err.println("MRC features computed: " + mrcFeatures.size());

		return counts;
	}

	/**
	 * Prints personality scores to standard output, and model details if
	 * required.
	 * 
	 * @param models
	 *            array of Weka models.
	 * @param scores
	 *            array of personality scores to print.
	 * @param modelIndex
	 *            index of the model used in the MODEL_NAMES array.
	 * @param printModels
	 *            if true, prints out a textual representation of the models.
	 * @param out
	 *            output stream.
	 */

	public void printOutput(Classifier[] models, double[] scores,
			int modelIndex, boolean printModels, boolean self, PrintStream out) {
		
		String adj = "Observed";
		if (self) { adj = "Self-assessed"; }
		
		out.println();
		out.println();
		String header = "Output of " + MODEL_NAMES[modelIndex] + ":";
		out.println(header);
		out.println(header.replaceAll(".", "-"));
		out.println();
		for (int j = 0; j < models.length; j++) {
			if (printModels) {
				out.println();
				out.println("--------------");
				out.println();
				out.println(adj + " " + DIMENSIONS[j].toLowerCase() + " " + models[j].toString());
				out.println();
			}
			
		}
		
		out.println();
		out.println();
		
		for (int j = 0; j < models.length; j++) {
			String spaces = "                              "
			.substring(DIMENSIONS[j].length());
	
			out.println(adj + " " + DIMENSIONS[j].toLowerCase() + " score: " + spaces + weka.core.Utils.doubleToString(scores[j],3));
		}
		
		out.println();
		out.println("Models are trained to output scores on a scale from 1 (low) to 7 (high),"+ LS + 
				"the scores might need to be normalized depending on the application domain." + LS + 
				"You can use models relative to your application domain and improve accuracies " + LS + 
				"by running the recognizer on multiple files using the -d switch.");
	
	}
	
	
	
	
	/**
	 * Prints personality scores of multiple files to standard output, and model details if
	 * required.
	 * 
	 * @param models
	 *            array of Weka models.
	 * @param scores
	 *            map associating each file to an array of personality scores to print.
	 * @param modelIndex
	 *            index of the model used in the MODEL_NAMES array.
	 * @param printModels
	 *            if true, prints out a textual representation of the models.
	 * @param out
	 *            output stream.
	 */

	public void printOutput(Classifier[] models, Map<File,Double[]> scores,
			int modelIndex, boolean printModels, boolean self, PrintStream out) {
		
		String adj = "Observed";
		if (self) { adj = "Self-assessed"; }
		
		out.println();
		out.println();
		String header = "Output of " + MODEL_NAMES[modelIndex] + ":";
		out.println(header);
		out.println(header.replaceAll(".", "-"));
		out.println();
		
		for (int j = 0; j < models.length; j++) {
			if (printModels) {
				out.println();
				out.println("--------------");
				out.println();
				out.println(adj + " " + DIMENSIONS[j].toLowerCase() + " " + models[j].toString());
				out.println();
			}
		}
		
		out.println();
		out.println();
		out.println("Estimates of " + adj.toLowerCase() +  " personality for each file, using standardized features:");
		// out.println("--------------------------------------------------------" + "------------------------".substring(0, adj.length()));
		out.println();
		out.print("File" + "              " + "\t");
		for (int j = 0; j < models.length; j++) {
			out.print(DIMENSIONS[j].substring(0,5));
			if (j < models.length - 1) { out.print("\t"); } else { out.println(); }
		}
			
		for(File file : scores.keySet()) {
			out.print(file.getName() + "                 ".substring(Math.min(16,file.getName().length())) + "\t");
			for (int j = 0; j < models.length; j++) {
				out.print(weka.core.Utils.doubleToString(scores.get(file)[j], 3));
				if (j < models.length - 1) { out.print("\t"); } else { out.println(); }
			}
		}
		out.println();
		out.println();
		for (String dim : DIMENSIONS) { 
			out.println(dim.substring(0,5) + " = " + dim);
		}
		out.println();
		out.println("Models are trained to output scores on a scale from 1 (low) to 7 (high),"+ LS + 
				"the scores might need to be normalized depending on the application domain.");
		
	}
	
	


	/**
	 * Loads a Weka model in memory, from a file saved through the Weka GUI.
	 * JRE/JDK and Weka versions need to be the same or compatible when saving
	 * the models and loading them.
	 * 
	 * @param modelFile
	 *            saved Weka model.
	 * @return Weka model object.
	 * @throws Exception
	 */
	private Classifier loadWekaModel(File modelFile) throws Exception {

		System.err.println("Loading model " + modelFile.getAbsolutePath()
				+ "...");
		InputStream is = new FileInputStream(modelFile);
		ObjectInputStream objectInputStream = new ObjectInputStream(is);
		Classifier classifier = (Classifier) objectInputStream.readObject();
		objectInputStream.close();

		// String className = classifier.getClass().getName();
		// System.err.println("class: " + className);
		return classifier;
	}



	/**
	 * Computes the average value of MRC features for all words in the text.
	 * Ratio is computed over the words with a non-zero value. Does not check
	 * for the word PoS, a word is associated with the feature value of the
	 * first homonym with a PoS in the MRC_POS array.
	 * 
	 * @param db
	 *            mapping associating each word with a line of the MRC Psycholinguistic Database.
	 * @param text
	 *            input text.
	 * @return mapping associating each 14 MRC feature name with the average
	 *         value of that feature for all words with non null values in the
	 *         input text (Double objects).
	 */
	private Map<String,Double> getMRCCounts(MRCDatabase db, String text) throws QueryException {
		
		// tokenize text
		String[] words = LIWCDictionary.tokenize(text);

		Map<String,Double> counts = new LinkedHashMap<String, Double>(MRC_FEATURES.length);	
		Map<String,Integer> nonzeroWords = new LinkedHashMap<String, Integer>(MRC_FEATURES.length);

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

	/**
	 * Runs a Weka model on the input feature (attribute) counts and returns the
	 * model's score.
	 * 
	 * @param model
	 *            Weka model to use for regression. It must have been trained
	 *            using the same attribute set as the keys of the userData
	 *            hashtable, plus the class attribute. The class must be
	 *            numeric.
	 * @param userData
	 *            test instance, as a hashtable associating attribute strings of
	 *            the model with attribute values (Double objects).
	 * @return numeric output of the model.
	 */
	private double runWekaModel(Classifier model, Map<String,Double> userData)
			throws Exception {

		// create new instance from test data
		Instance userInst = new SparseInstance(userData.size() + 1);
		// create empty dataset
		Instances dataset = new Instances(new BufferedReader(new FileReader(
				attributeFile)), 1);
		if (userData.size() < dataset.numAttributes() - 1) { dataset.deleteAttributeAt(dataset.attribute("WC").index()); }
		
		userInst.setDataset(dataset);
		dataset.setClassIndex(dataset.numAttributes() - 1);

		for (Attribute attr : Collections.list((Enumeration<Attribute>) dataset.enumerateAttributes())) {

			if (userData.containsKey(attr.name().toUpperCase())) {
				if (userData.get(attr.name().toUpperCase()).toString().equals(
						"?")) {
					userInst.setMissing(attr);
					System.err.println("Warning: attribute " + attr.name()
							+ " missing");
				} else {
					double attrValue = userData.get(attr.name()
							.toUpperCase());
					userInst.setValue(attr, attrValue);
				}

			} else {
				System.err.println("No value for feature " + attr.name()
						+ ", setting as missing value...");
				userInst.setMissing(attr);
			}
		}
		userInst.setClassMissing();

		// run model for test data
		double result = model.classifyInstance(userInst);
		return result;
	}
	
	
	/**
	 * Runs the models of each personality trait for each file in the directory.
	 * Feature values are standardized.
	 * 
	 * @param dir input directory containing multiple text files.
	 * @param models models of each Big Five personality trait.
	 * @param outputArffFile Weka <code> arff</code> file to print the feature values and scores to (null=none).
	 * @return map associating each file with an array of personality scores for 
	 * each trait.
	 */
	public Map<File,Double[]> computeScoresOverCorpus(File dir, Classifier[] models, File outputArffFile) {
		
		// load dataset
		Map<File,Instance> dataset = getCorpusDataset(dir);
		
		Instances inputDataset = new Instances(dataset.get(dataset.keySet().iterator().next()).dataset());
		// create full dataset with features and scores
		FastVector fv = new FastVector(inputDataset.numAttributes()+DIMENSIONS.length);
		fv.addElement(new Attribute("filename", (FastVector) null));
		for (int i = 0; i < inputDataset.numAttributes() - 1; i++)
		{
			fv.addElement(new Attribute(inputDataset.attribute(i).name()));
		}
		for (String dim : DIMENSIONS) { 
			fv.addElement(new Attribute(dim));
		}
		Instances outputArff = new Instances("features_"+dir.getAbsolutePath(), fv, dataset.size());
		
		
		Map<File,Double[]> map = new LinkedHashMap<File, Double[]>(dataset.size());
		
		// compute models
		for (File file : dataset.keySet()) {
			
			Instance inst = new SparseInstance(fv.size());
			inst.setDataset(outputArff);
			inst.setValue(0, file.getAbsolutePath());
			// copy feature values to arff file, but not score
			for (int i = 0; i < dataset.get(file).numAttributes()-1; i++) { inst.setValue(i+1, dataset.get(file).value(i)); }
						
			
			Double[] scores = new Double[models.length];
			for (int i = 0; i < models.length; i++) {
				try {
					scores[i] = models[i].classifyInstance(dataset.get(file));
					inst.setValue(dataset.get(file).numAttributes()+i, scores[i]);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			map.put(file, scores);
			outputArff.add(inst);
		}
		
		// write feature+score in arff file if option is specified
		if (outputArffFile != null) {
			try {
						FileWriter fw = new FileWriter(outputArffFile);
						fw.write(outputArff.toString());
						fw.close();				
						System.err.println("Features and scores written in arff file " + outputArffFile.getAbsolutePath());
				} catch (Exception e) { e.printStackTrace(); }
		}
		
		return map;
		
	}
	
	/**
	 * Computes the features for each file in the directory, and 
	 * returns a dataset with one instance per text file, and a missing class.
	 * 
	 * @param dir input directory containing text files to analyse.
	 * @return a Weka dataset with one instance per text file, and a missing class.
	 */
	private Map<File,Instance> 	getCorpusDataset(File dir) {
		
		Map<File,Instance> map = new LinkedHashMap<File, Instance>();
		Instances dataset = null; 
		try {
		dataset = new Instances(new BufferedReader(new FileReader(
				attributeFile)), 1);
		dataset.setClassIndex(dataset.numAttributes() - 1);
	
		for (File file : dir.listFiles()) {
			System.err.println();
			System.err.println("Computing features for file " + file.getName() + "...");
			String text = Utils.readFile(file);		
			Map<String,Double> counts = getFeatureExtractionFromText(text, false);
			// add one instance to dataset
			addInstance(dataset, counts);
		}
			
		System.err.println("Computing standardized values for each feature over the whole corpus ("+ dataset.numInstances() + " files)");
		Standardize stdFilter = new Standardize();
		stdFilter.setInputFormat(dataset);
		dataset = Filter.useFilter(dataset, stdFilter);
		
		int c = 0;
		for (File file : dir.listFiles()) {
			map.put(file, dataset.instance(c));
			c++;
		}


		} catch (Exception e) {
			e.printStackTrace();
		}

		return map;
	}


	
	/**
	 * Adds an instance to the dataset, with features defined by the map
	 * in the argument. The class of the instance is set to missing.
	 * 
	 * @param dataset Weka dataset.
	 * @param counts map associating feature names to counts. The features
	 * must be identical to those defined in the dataset header.
	 */
	private Instance addInstance(Instances dataset, Map<String, Double> counts) {
		
		try {
				
		// create new instance from test data
		Instance userInst = new SparseInstance(counts.size() + 1);

		userInst.setDataset(dataset);

		for (Attribute attr : Collections.list((Enumeration<Attribute>) dataset.enumerateAttributes())) {

			if (counts.containsKey(attr.name().toUpperCase())) {
				if (counts.get(attr.name().toUpperCase()).toString().equals(
						"?")) {
					userInst.setMissing(attr);
					System.err.println("Warning: attribute " + attr.name()
							+ " missing");
				} else {
					double attrValue = counts.get(attr.name()
							.toUpperCase());
					userInst.setValue(attr, attrValue);
				}

			} else {
				System.err.println("Warning: feature " + attr.name()
						+ " has no value, setting as missing...");
				userInst.setMissing(attr);
			}
		}
				
		userInst.setClassMissing();
		dataset.add(userInst);
		
		return userInst;
		
		} catch (Exception e) { e.printStackTrace(); }
		return null;
	}
	
	/**
	 * Returns a mapping associating features in the LIWC.CAT file to shortcuts used in the 
	 * Weka models.
	 * 
	 * @return mapping asssociating long names in the LIWC dictionary to the short names in the
	 * Weka models. 
	 */
	private Map<String,String> getShortFeatureNames() {
		
		Map<String,String> shortcuts = new LinkedHashMap<String,String>();
		
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



}