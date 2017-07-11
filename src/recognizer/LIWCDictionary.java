package recognizer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Interface to the LIWC dictionary, implementing patterns for each LIWC category
 * based on the LIWC.CAT file.
 *  
 * @author Francois Mairesse, <a href=http://www.mairesse.co.uk
 *         target=_top>http://www.mairesse.co.uk</a>
 * @version 1.01
 */
public class LIWCDictionary {

	/** Mapping associating LIWC features to regular expression patterns. */
	private Map<String,Pattern> map;
	
	/**
	 * Loads dictionary from LIWC dictionary tab-delimited text file (with
	 * variable names as first row). Each word category is converted into a
	 * regular expression that is a disjunction of all its members.
	 * 
	 * @param catFile
	 *            dictionary file, it should be pointing to the LIWC.CAT file of
	 *            the Linguistic Inquiry and Word Count software (Pennebaker &
	 *            Francis, 2001).
	 */
	public LIWCDictionary(File catFile) {
		try {
			map = loadLIWCDictionary(catFile);
			System.err.println("LIWC dictionary loaded ("
					+ map.size() + " lexical categories)");

		} catch (IOException e) {
			System.err.println("Error: file " + catFile + " doesn't exist");
			e.printStackTrace();
			System.exit(1);
		} catch (NullPointerException e) {
			System.err.println("Error: LIWC dicitonary file " + catFile + " doesn't have the right format");
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	/**
	 * Loads dictionary from LIWC dictionary tab-delimited text file (with
	 * variable names as first row). Each word category is converted into a
	 * regular expression that is a disjunction of all its members.
	 * 
	 * @param dicFile
	 *            dictionary file, it should be pointing to the LIWC.CAT file of
	 *            the Linguistic Inquiry and Word Count software (Pennebaker &
	 *            Francis, 2001).
	 * @return hashtable associating each category with a regular expression
	 *         (Pattern object) matching each word.
	 */
	private Map<String,Pattern> loadLIWCDictionary(File dicFile) throws IOException {

		BufferedReader reader = new BufferedReader(new FileReader(dicFile));
		String line;

		Map<String,Pattern> wordLists = new LinkedHashMap<String,Pattern>();
		String currentVariable = "";
		String catRegex = "";
		int word_count = 0;

		while ((line = reader.readLine()) != null) {

			// if encounter new variable
			if (line.matches("\\t[\\w ]+")) {
				// add full regex to database
				if (!catRegex.equals("")) {
					catRegex = catRegex.substring(0, catRegex.length() - 1);
					catRegex = "(" + catRegex + ")";
					catRegex = catRegex.replaceAll("\\*", "[\\\\w']*");
					wordLists.put(currentVariable, Pattern.compile(catRegex));
				}
				// update variable
				currentVariable = line.split("\t")[1];
				catRegex = "";

			} else if (line.matches("\t\t.+ \\(\\d+\\)")) {
				word_count++;
				String newPattern = line.split("\\s+")[1].toLowerCase();
				catRegex += "\\b" + newPattern + "\\b|";
			}
		}
		//  add last regex to database
		if (!catRegex.equals("")) {
			catRegex = catRegex.substring(0, catRegex.length() - 1);
			catRegex = "(" + catRegex + ")";
			catRegex = catRegex.replaceAll("\\*", "[\\\\w']*");
			wordLists.put(currentVariable, Pattern.compile(catRegex));
		}

		reader.close();

		// System.err.println(word_count + " words and " + wordLists.size() +"
		// categories loaded in LIWC dictionary");
		return wordLists;
	}
	
	
	/**
	 * Returns a map associating each LIWC categories to the number of
	 * their occurences in the input text. The counts are computed matching
	 * patterns loaded. It doesn't produce punctuation counts.
	 * 
	 * @param text input text.
	 * @param absoluteCounts includes counts that aren't relative to the total word 
	 * count (e.g. actual word count).
	 * @return hashtable associating each LIWC category with the percentage of
	 *         words in the text belonging to it.
	 */
	public Map<String,Double> getCounts(String text, boolean absoluteCounts) {

		Map<String,Double> counts = new LinkedHashMap<String, Double>(map.size());
		String[] words = tokenize(text);
		String[] sentences = splitSentences(text);
		
		System.err.println("Input text splitted into " + words.length
				+ " words and " + sentences.length + " sentences");
		
		// word count (NOT A PROPER FEATURE)
		if (absoluteCounts) { counts.put("WC", new Double(words.length)); }
		counts.put("WPS", new Double(1.0 * words.length / sentences.length));
		
		// type token ratio, words with more than 6 letters, abbreviations,
		// emoticons, numbers
		int sixletters = 0;
		int numbers = 0;
		for (int i = 0; i < words.length; i++) {
			String word = words[i].toLowerCase();
			if (word.length() > 6) {
				sixletters++;
			}

			if (word.matches("-?[,\\d+]*\\.?\\d+")) {
				numbers++;
			}
		}
		
		Set<String> types = new LinkedHashSet<String>(Arrays.asList(words));
		counts.put("UNIQUE", new Double(100.0 * types.size() / words.length));
		counts.put("SIXLTR", new Double(100.0 * sixletters / words.length));
		// abbreviations
		int abbrev = Utils.countMatches("\\w\\.(\\w\\.)+", text);
		counts.put("ABBREVIATIONS", new Double(100.0 * abbrev / words.length));
		// emoticons
		int emoticons = Utils
				.countMatches("[:;8%]-[\\)\\(\\@\\[\\]\\|]+", text);
		counts.put("EMOTICONS", new Double(100.0 * emoticons / words.length));
		// text ending with a question mark
		int qmarks = Utils.countMatches("\\w\\s*\\?", text);
		counts.put("QMARKS", new Double(100.0 * qmarks / sentences.length));
		// punctuation
		int period = Utils.countMatches("\\.", text);
		counts.put("PERIOD", new Double(100.0 * period / words.length));
		int comma = Utils.countMatches(",", text);
		counts.put("COMMA", new Double(100.0 * comma / words.length));
		int colon = Utils.countMatches(":", text);
		counts.put("COLON", new Double(100.0 * colon / words.length));
		int semicolon = Utils.countMatches(";", text);
		counts.put("SEMIC", new Double(100.0 * semicolon / words.length));
		int qmark = Utils.countMatches("\\?", text);
		counts.put("QMARK", new Double(100.0 * qmark / words.length));
		int exclam = Utils.countMatches("!", text);
		counts.put("EXCLAM", new Double(100.0 * exclam / words.length));
		int dash = Utils.countMatches("-", text);
		counts.put("DASH", new Double(100.0 * dash / words.length));
		int quote = Utils.countMatches("\"", text);
		counts.put("QUOTE", new Double(100.0 * quote / words.length));
		int apostr = Utils.countMatches("'", text);
		counts.put("APOSTRO", new Double(100.0 * apostr / words.length));
		int parent = Utils.countMatches("[\\(\\[{]", text);
		counts.put("PARENTH", new Double(100.0 * parent / words.length));
		int otherp = Utils.countMatches("[^\\w\\d\\s\\.:;\\?!\"'\\(\\{\\[,-]",
				text);
		counts.put("OTHERP", new Double(100.0 * otherp / words.length));
		int allp = period + comma + colon + semicolon + qmark + exclam + dash
				+ quote + apostr + parent + otherp;
		counts.put("ALLPCT", new Double(100.0 * allp / words.length));

		// PATTERN MATCHING

		// store word in dic
		boolean[] indic = new boolean[words.length];
		for (int i = 0; i < indic.length; i++) {
			indic[i] = false;
		}

		// first get all lexical counts
		for (String cat: map.keySet()) {

			// add entry to output hash
			Pattern catRegex = map.get(cat);
			int catCount = 0;

			for (int i = 0; i < words.length; i++) {

				String word = words[i].toLowerCase();
				Matcher m = catRegex.matcher(word);
				while (m.find()) {
					catCount++;
					indic[i] = true;
				}
			}
			counts.put(cat, new Double(100.0 * catCount / words.length));
		}

		// put ratio of words matched
		int wordsMatched = 0;
		for (int i = 0; i < indic.length; i++) {
			if (indic[i]) {
				wordsMatched++;
			}
		}
		counts.put("DIC", new Double(100.0 * wordsMatched / words.length));
		// add numerical numbers
		double nonNumeric = ((Double) counts.get("NUMBERS")).doubleValue();
		counts.put("NUMBERS", new Double(nonNumeric + 100.0 * numbers
				/ words.length));

		return counts;
	}

	
	/**
	 * Splits a text into words separated by non-word characters.
	 * 
	 * @param text text to tokenize.
	 * @return an array of words.
	 */
	public static String[] tokenize(String text) {

		//deletes all non characters and reduce spaces more than ones. then split the text based on one space
		String words_only = text.replaceAll("\\W+\\s*", " ").replaceAll(
				"\\s+$", "").replaceAll("^\\s+", "");
		String[] words = words_only.split("\\s+");
		return words;
	}
	
	
	
	/**
	 * Splits a text into sentences separated by a dot, exclamation point or question mark.
	 * 
	 * @param text text to tokenize.
	 * @return an array of sentences.
	 */
	public static String[] splitSentences(String text) {
	
		return text.split("\\s*[\\.!\\?]+\\s+");
	}
}
