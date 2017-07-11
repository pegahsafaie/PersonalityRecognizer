package recognizer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.Hashtable;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/*
 * Created on Sep 4, 2006
 *
 */

/**
 * Library of program-independent static methods.
 * 
 * @author Francois Mairesse, <a href=http://www.mairesse.co.uk
 *         target=_top>http://www.mairesse.co.uk</a>
 */
public class Utils {


	/**
	 * Counts the number of times a pattern appears in a string.
	 * 
	 * @param regex
	 *            regular expression string to be matched, it must be in the
	 *            appropriate format to be compiled into a Pattern object.
	 * @param text
	 *            input text.
	 * @return number of matches found.
	 */
	public static int countMatches(String regex, String text) {
		Pattern p = Pattern.compile(regex);
		Matcher m = p.matcher(text);
		int matches = 0;
		while (m.find()) {
			matches++;
		}
		return matches;
	}

	/**
	 * Prints the content of a hashtable containing strings to the standard
	 * output.
	 * 
	 * @param ht
	 *            hashtable with string keys and values that can be represented
	 *            using their toString() method.
	 * @param out
	 *            output stream.
	 */
	public static void printHash(Hashtable<? extends Object,? extends Object> ht, PrintStream out)
			throws Exception {

		for (Object key : ht.keySet()) {
			out.println(key.toString() + " " + ht.get(key).toString());
		}
	}
	
	/**
	 * Prints the content of a Map containing strings to the standard
	 * output.
	 * 
	 * @param map
	 *            map with string keys and values that can be represented
	 *            using their toString() method.
	 * @param out
	 *            output stream.
	 */
	public static void printMap(Map<? extends Object,? extends Object> map, PrintStream out)
			throws Exception {

		for (Object key : map.keySet()) {
			out.println(key.toString() + " " + map.get(key).toString());
		}
	}	

	/**
	 * Reads the content of a text file into a string.
	 * 
	 * @param file
	 *            text file.
	 * @return string containing the text, with a line separator character
	 *         between lines.
	 * @throws Exception
	 */
	public static String readFile(File file) throws Exception {

		BufferedReader br = new BufferedReader(new FileReader(file));

		String line = "";
		String output = "";
		while ((line = br.readLine()) != null) {
			output += line + System.getProperty("line.separator");
		}
		br.close();
		return output;
	}

	/**
	 * Reads the content of all files in a directory into a single string.
	 * 
	 * @param dir
	 *            directory containing text files.
	 * @return string containing the text, with a line separator character
	 *         between lines.
	 * @throws Exception
	 */
	public static String readDir(File dir) throws Exception {

		String output = "";
		File[] files = dir.listFiles();
		for (int i = 0; i < files.length; i++) {
			if (files[i].isFile()) {
				output += readFile(files[i]);
			}
		}
		return output;
	}
}
