/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.integration.tests.model_zoo;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooProvider;
import ai.djl.util.Utils;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.ServiceLoader;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class ModelZooTest {

    @BeforeClass
    public void setUp() {
        // force downloading without cache in .djl.ai folder.
        System.setProperty("DJL_CACHE_DIR", "build/cache");
    }

    @AfterClass
    public void tearDown() {
        System.setProperty("DJL_CACHE_DIR", "");
    }

    @Test
    public void testDownloadModels() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || Boolean.getBoolean("offline")) {
            return;
        }

        ServiceLoader<ZooProvider> providers = ServiceLoader.load(ZooProvider.class);
        for (ZooProvider provider : providers) {
            ModelZoo zoo = provider.getModelZoo();
            if (zoo != null) {
                for (ModelLoader<?, ?> modelLoader : zoo.getModelLoaders()) {
                    List<Artifact> artifacts = modelLoader.listModels();
                    for (Artifact artifact : artifacts) {
                        Model model = modelLoader.loadModel(artifact.getProperties());
                        model.close();
                    }
                }
            }
            Utils.deleteQuietly(Paths.get("build/cache"));
        }
    }
}
