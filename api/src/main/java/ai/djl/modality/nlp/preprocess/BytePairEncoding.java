package ai.djl.modality.nlp.preprocess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 */
public class BytePairEncoding implements TextProcessor {

    /**
     * Unlike the original paper, we use the NUL char U+0000 to indicate the end of a word.
     * It allows for easier checks and is very unlikely to occur in real text.
     */
    public static char END_OF_WORD = 0;

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
     * @param createTokens if true, non-existent tokens are automatically created. If false, a
     *     runtime exception is thrown if a token in the splitting does not exist.
     */
    public void addSplit(final String word, final List<String> splitting, final boolean createTokens) {
        for (final String token : splitting) {
            if (getTokenId(token) == null) {
                if (createTokens) {
                    addToken(token);
                } else {
                    throw new IllegalArgumentException("Cannot add splitting for word '" + word + "' using token '" +
                            token + "', token is not in the dictionary.");
                }
            }
        }
        this.splitLookup.put(word, splitting);
    }

    @Override
    public List<String> preprocess(List<String> tokens) {
        return null;
    }

    static class BytePairEncodingTrainer {
        /** A mapping from tokens to their frequency. */
        private Map<String, Long> tokenToFrequency = new HashMap<>();

        private final BytePairEncoding bpEncoding;

        public BytePairEncodingTrainer(final BytePairEncoding bpEncoding) {
            this.bpEncoding = bpEncoding;
        }

        public void startEpoch() {
            tokenToFrequency.clear();
        }

        private ArrayList<String> createInitialTokens(final String word) {
            return word.codePoints().map((codepoint) -> new String(Character.toChars(codepoint)) );
        }

        public void trainEncoding(final Map<String, Long> wordsToFrequencies, final int targetSize) {
            //setup: create a local copy of the words -> frequencies map, make sure words are ended by EOW token.
            final Map<List<String>, Long> tokenizedWordsToFrequencies =



            while (bp) {

            }
        }

    }

    // Input: words and their frequencies

    // Initial state: tokenize into codepoints, add special EOW token (we use the NUL codepoint)

    // a split rule table, keys are merged tokens, values are pairs of result tokens

    // getting a (potentially unknown) tokenization: Iteratively look up larger tokens starting
    // left-to-right,
    // if we find one, split into left, center, right and continue tokenization

}
