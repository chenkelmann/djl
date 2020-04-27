package ai.djl.modality.nlp.preprocess;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BytePairEncoding {

    /** A set of all tokens and their dictionary ids. */
    private Map<String, Long> tokenToId = new HashMap<>();

    /** This Lookup table speeds up preprocessing of words. */
    private Map<String, List<String>> splitLookup = new HashMap<>();

    /**
     * Looks up the ID for a token.
     *
     * @param token the token to look up
     * @return null if the token does not exist, its id otherwise
     */
    public Long getTokenId(final String token) {
        return tokenToId.get(token);
    }

    /**
     * Adds a token to the underlying token mapping. If the token already exists, nothing changes.
     *
     * @param token A new or existing non-null token.
     * @return the existing or new Id for the given token.
     */
    public long addToken(final String token) {
        final Long existingId = getTokenId(token);
        if (existingId != null) {
            return existingId;
        } else {
            final long newId = tokenToId.size();
            tokenToId.put(token, newId);
            return newId;
        }
    }

    /**
     * Adds a tokenization split for the given word.
     *
     * @param word A word, must end with the NUL char
     * @param splitting A splitting of the given word. Must concatenate back to the given word.
     * @param createTokens if true, non-existant tokens are automatically created. If false, a
     *     runtime exception is thrown if a token in the splitting does not exist.
     */
    public void addSplit(
            final String word, final List<String> splitting, final boolean createTokens) {}

    public static BytePairEncoding trainEncoding(final Map<String, Long> wordsToFrequencies) {
        // first, determine all codepoints present in the words, turn them into the base dictionary

        // then, create
        return null;
    }

    /** A mapping from tokens to their frequency. */
    private Map<String, Long> tokenToFrequency = new HashMap<>();

    // Input: words and their frequencies

    // Initial state: tokenize into codepoints, add special EOW token (we use the NUL codepoint)

    // a split rule table, keys are merged tokens, values are pairs of result tokens

    // getting a (potentially unknown) tokenization: Iteratively look up larger tokens starting
    // left-to-right,
    // if we find one, split into left, center, right and continue tokenization

}
