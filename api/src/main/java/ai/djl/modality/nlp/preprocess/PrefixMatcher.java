package ai.djl.modality.nlp.preprocess;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Helper class to efficiently find prefixes in Strings.
 * Internally operates on codepoints to prevent mismatches for prefixes containing glyphs outside
 * the Basic Multilingual Plane (BMP).
 * Uses a Trie to make Lookups as efficient as possible, runtime is O(min(M, N)) where M is the
 * longest prefix length and N the length of the query string.
 */
public class PrefixMatcher {

    /**
     * The root of this trie.
     */
    private final TrieNode trieRoot = new TrieNode(0);

    /**
     * Creates a prefix matcher matching the given strings.
     * @param prefixes the prefixes to match, elements must not be null but need not be unique
     */
    public PrefixMatcher(final Collection<String> prefixes) {
        prefixes.forEach(this::insertPrefix);
    }

    /**
     * Inserts the given prefix into the trie.
     * @param prefix the prefix to insert
     */
    private void insertPrefix(final String prefix) {
        TrieNode current = trieRoot;
        int idx = 0;
        while (idx < prefix.length()) {
            final int cp = prefix.codePointAt(idx);
            final int cpCharCount = Character.charCount(cp);
            current = current.children.computeIfAbsent(cp, (ignored) -> new TrieNode(cpCharCount));
            idx += cpCharCount;
        }
        current.word = prefix;
    }

    /**
     * Searches the given string for the longest possible prefix defined in this matcher.
     * Assuming this matcher contains "f", "foo" and "baz" it will yield the following:
     * "none", 0 -> ""
     * "false",0 -> "f"
     * "false",1 -> ""
     * "bar", 0 -> ""
     * "fofoo", 0 -> "f"
     * "fofoo", 1 -> ""
     * "fofoo", 2 -> "foo"
     * "bazbaz", 0 -> "baz"
     * @param s the string to search, not null
     * @param offset the offset (in chars) into the String, must be > 0, but may be outside the
     *               range of the string length
     * @return the matched prefix, may be the empty string, never null
     */
    public String findLongestPrefix(final String s, final int offset) {
        if (offset < 0) {
            throw new IllegalArgumentException("Illegal String offset: " + offset);
        }
        TrieNode current = trieRoot;
        int idx = offset;
        String lastWord = "";
        while (idx < s.length() && current != null) {
            final int cp = s.codePointAt(idx);
            current = current.children.get(cp);
            if (current != null && current.word != null) {
                lastWord = current.word;
            }
            idx += Character.charCount(cp);
        }
        return lastWord;
    }

    /**
     * Helper class to build the trie with.
     */
    private static class TrieNode {
        private final Map<Integer, TrieNode> children = new HashMap<>();
        private final int charCount;
        private String word = null;

        /**
         * Builds a node representing a unicode code point with the given number of chars
         * @param charCount number of chars in the codepoint, >= 0
         */
        private TrieNode(final int charCount) {
            this.charCount = charCount;
        }

        @Override
        public String toString() {
            return "{" + children.size() + ", " + charCount + ", " + word + "}";
        }
    }
}
