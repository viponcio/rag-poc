const express = require("express");
const app = express();
const { Anthropic } = require("@anthropic-ai/sdk");
const { createClient } = require("@supabase/supabase-js");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { CheerioWebBaseLoader } = require("langchain/document_loaders/web/cheerio");
const crypto = require('crypto');
require("dotenv").config();

app.use(express.json());

// Create Anthropic client
const anthropic = new Anthropic(process.env.ANTHROPIC_API_KEY);

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

// Function to generate deterministic 1536-dimension embeddings
function generateDeterministicEmbedding(text) {
  // Create a hash from the text
  const hash = crypto.createHash('sha256').update(text).digest('hex');
  
  // Use the hash to seed a deterministic random number generator
  const embedding = new Array(1536);
  
  // Generate 1536 values in range [-1, 1]
  for (let i = 0; i < 1536; i++) {
    // Create a deterministic value based on the hash and position
    const seed = parseInt(hash.substring((i % 32) * 2, (i % 32) * 2 + 8), 16);
    const value = (seed / 0xffffffff) * 2 - 1; // Convert to range [-1, 1]
    embedding[i] = value;
  }
  
  // Normalize the embedding to unit length
  const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  const normalizedEmbedding = embedding.map(val => val / magnitude);
  
  return normalizedEmbedding;
}

app.post("/embed", async (req, res) => {
  try {
    await generateAndStoreEmbeddings();
    res.status(200).json({ message: "Successfully Embedded" });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      message: "Error occurred",
      error: error.toString()
    });
  }
});

app.post("/query", async (req, res) => {
  try {
    const { query } = req.body;
    const result = await handleQuery(query);
    res.status(200).json(result);
  } catch (error) {
    console.log(error);
    res.status(500).json({
      message: "Error occurred",
      error: error.toString()
    });
  }
});

async function generateAndStoreEmbeddings() {
    const loader = new CheerioWebBaseLoader(
      "https://www.inboxpurge.com/faq"
    );
    const docs = await loader.load();

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const chunks = await textSplitter.splitDocuments(docs);

    const promises = chunks.map(async (chunk) => {
      const cleanChunk = chunk.pageContent.replace(/\n/g, " ");

      // Generate deterministic 1536-dimension embeddings
      const embedding = generateDeterministicEmbedding(cleanChunk);
      
      // Ensure no null values
      if (embedding.some(val => val === null || val === undefined || Number.isNaN(val))) {
        throw new Error("Invalid embedding generated with null or undefined values");
      }

      const { error } = await supabase.from("doc").insert({
        content: cleanChunk,
        embedding,
      });

      if (error) {
        console.error("Supabase insertion error:", error);
        throw error;
      }
    });

    await Promise.all(promises);
}

async function handleQuery(query) {
  const input = query.replace(/\n/g, " ");
  
  // Generate embedding for the query
  const embedding = generateDeterministicEmbedding(input);

  const { data: doc, error } = await supabase.rpc("match_documents", {
    query_embedding: embedding,
    match_threshold: 0.5,
    match_count: 10,
  });

  if (error) throw error;

  let contextText = "";

  contextText += doc
    .map((document) => `${document.content.trim()}---\n`)
    .join("");

  // Using Anthropic's chat API
  // Replace your current message creation code with:
const message = await anthropic.beta.messages.create({
  model: "claude-3-sonnet-20240229", // Updated model name
  max_tokens: 1024,
  messages: [
    {
      role: "user",
      content: `Context sections: "${contextText}" Question: "${query}" Answer as simple text:`,
    },
  ],
  temperature: 0.8,
});

  return message.content[0].text;
}

app.listen("3035", () => {
  console.log("App is running on port 3035");
});