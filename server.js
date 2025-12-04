// server.js
const express = require("express");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

// Simple chatbot logic
app.post("/chat", (req, res) => {
  const userMessage = req.body.message?.toLowerCase();
  let reply = "";

  if (userMessage === "hi") {
    reply = "hi i am bot";
  } else if (userMessage === "hello") {
    reply = "how can i help you";
  } else if (userMessage === "bye") {
    reply = "bye";
  } else {
    reply = "I don't understand.";
  }

  res.json({ reply });
});

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
