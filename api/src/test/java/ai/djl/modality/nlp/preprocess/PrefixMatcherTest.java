/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.modality.nlp.preprocess;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;

public class PrefixMatcherTest {
    @Test
    public void testMatching() {
        PrefixMatcher matcher = new PrefixMatcher(Arrays.asList("f", "foo", "baz"));
        Assert.assertEquals(matcher.findLongestPrefix("", 0), "");
        Assert.assertEquals(matcher.findLongestPrefix("", 1), "");
        Assert.assertEquals(matcher.findLongestPrefix("none", 0), "");
        Assert.assertEquals(matcher.findLongestPrefix("false", 0), "f");
        Assert.assertEquals(matcher.findLongestPrefix("false", 1), "");
        Assert.assertEquals(matcher.findLongestPrefix("bar", 0), "");
        Assert.assertEquals(matcher.findLongestPrefix("barbaz", 3), "baz");
        Assert.assertEquals(matcher.findLongestPrefix("fofoo", 0), "f");
        Assert.assertEquals(matcher.findLongestPrefix("fofoo", 1), "");
        Assert.assertEquals(matcher.findLongestPrefix("fofoo", 2), "foo");
        Assert.assertEquals(matcher.findLongestPrefix("bazbaz", 0), "baz");
        Assert.assertEquals(matcher.findLongestPrefix("baz_baz", 4), "baz");
        Assert.assertEquals(matcher.findLongestPrefix("baz_baz_", 4), "baz");
    }
}
