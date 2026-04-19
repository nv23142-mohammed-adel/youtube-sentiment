// popup.js

document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = 'AIzaSyARsLFPnLE2ozsNpbzEuZLqtvFeJKZpq9I';
  // const API_URL = 'http://my-elb-2062136355.us-east-1.elb.amazonaws.com:80';
  const API_URL = 'http://localhost:5000';
  // const API_URL = 'http://23.20.221.231:8080/'; We'll modify once backend is ready

  // ---------------------------------------------------------------------------
  // chrome.storage.local helpers (defensive — return null/no-op if unavailable)
  // ---------------------------------------------------------------------------

  function getLocalCache(videoId) {
    if (!chrome.storage?.local) return Promise.resolve(null);
    return new Promise(resolve => {
      chrome.storage.local.get(videoId, data => {
        if (chrome.runtime.lastError) { resolve(null); return; }
        resolve(data[videoId] || null);
      });
    });
  }

  function setLocalCache(videoId, data) {
    if (!chrome.storage?.local) return Promise.resolve();
    return new Promise(resolve => {
      chrome.storage.local.set({ [videoId]: data }, () => resolve());
    });
  }

  function clearLocalCache(videoId) {
    if (!chrome.storage?.local) return Promise.resolve();
    return new Promise(resolve => chrome.storage.local.remove(videoId, resolve));
  }

  // ---------------------------------------------------------------------------
  // API call
  // ---------------------------------------------------------------------------

  async function analyzeVideo(videoId, comments) {
    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId, comments })
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Analysis failed');
      return result;
    } catch (err) {
      console.error('Error analyzing video:', err);
      outputDiv.innerHTML += `<p>Error: ${err.message}</p>`;
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  function renderResults(videoId, data, fromCache) {
    const { predictions, sentimentCounts, metrics, chartImage, trendImage, wordcloudImage } = data;

    const sentimentLabel = { "1": "Positive", "0": "Neutral", "-1": "Negative" };
    const sentimentClass  = { "1": "sentiment-positive", "0": "sentiment-neutral", "-1": "sentiment-negative" };
    const commentClass    = { "1": "comment-positive",   "0": "comment-neutral",   "-1": "comment-negative" };

    const cacheBar = fromCache
      ? `<div class="cache-bar"><span>Loaded from cache</span><button id="refresh-btn">Refresh</button></div>`
      : '';

    outputDiv.innerHTML = `
      ${cacheBar}
      <div class="section">
        <div class="section-title">Comment Analysis Summary</div>
        <div class="metrics-container">
          <div class="metric">
            <div class="metric-title">Total Comments</div>
            <div class="metric-value">${metrics.totalComments}</div>
          </div>
          <div class="metric">
            <div class="metric-title">Unique Commenters</div>
            <div class="metric-value">${metrics.uniqueCommenters}</div>
          </div>
          <div class="metric">
            <div class="metric-title">Avg Comment Length</div>
            <div class="metric-value">${metrics.avgWordLength} words</div>
          </div>
          <div class="metric">
            <div class="metric-title">Avg Sentiment Score</div>
            <div class="metric-value">${metrics.normalizedSentimentScore}/10</div>
          </div>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Sentiment Breakdown</div>
        <img src="data:image/png;base64,${chartImage}" alt="Sentiment chart">
      </div>

      <div class="section">
        <div class="section-title">Sentiment Trend Over Time</div>
        <img src="data:image/png;base64,${trendImage}" alt="Sentiment trend">
      </div>

      <div class="section">
        <div class="section-title">Comment Wordcloud</div>
        <img src="data:image/png;base64,${wordcloudImage}" alt="Word cloud">
      </div>

      <div class="section">
        <div class="section-title">Top 25 Comments with Sentiments</div>
        <ul class="comment-list">
          ${predictions.slice(0, 25).map((item, i) => `
            <li class="comment-item ${commentClass[item.sentiment] || ''}">
              <span>${i + 1}. ${item.comment}</span><br>
              <span class="comment-sentiment ${sentimentClass[item.sentiment] || ''}">
                ${sentimentLabel[item.sentiment] || item.sentiment}
              </span>
            </li>`).join('')}
        </ul>
      </div>`;

    if (fromCache) {
      document.getElementById('refresh-btn').addEventListener('click', async () => {
        await clearLocalCache(videoId);
        location.reload();
      });
    }
  }

  // ---------------------------------------------------------------------------
  // YouTube comment fetching
  // ---------------------------------------------------------------------------

  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 500) {
        const response = await fetch(
          `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`
        );
        const data = await response.json();
        if (data.items) {
          data.items.forEach(item => {
            const snippet = item.snippet.topLevelComment.snippet;
            comments.push({
              text: snippet.textOriginal,
              timestamp: snippet.publishedAt,
              authorId: snippet.authorChannelId?.value || 'Unknown'
            });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      outputDiv.innerHTML += "<p>Error fetching comments.</p>";
    }
    return comments;
  }

  // ---------------------------------------------------------------------------
  // Main flow
  // ---------------------------------------------------------------------------

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    try {
      const url = tabs[0]?.url || '';
      const match = url.match(/^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/);

      if (!match) {
        outputDiv.innerHTML = "<p>This is not a valid YouTube URL.</p>";
        return;
      }

      const videoId = match[1];

      // Show something immediately so the popup doesn't look broken
      outputDiv.innerHTML = `<p style="color:#aaa;font-size:12px;">Loading…</p>`;

      // 1. Try local cache first (no network required)
      const cached = await getLocalCache(videoId);
      if (cached) {
        renderResults(videoId, cached, true);
        return;
      }

      // 2. Fetch YouTube comments
      outputDiv.innerHTML = `<p style="color:#aaa;font-size:12px;">Fetching comments for <code>${videoId}</code>…</p>`;

      const comments = await fetchComments(videoId);
      if (comments.length === 0) {
        outputDiv.innerHTML += "<p>No comments found for this video.</p>";
        return;
      }

      // 3. Run analysis on the server (server may also return a cached result)
      outputDiv.innerHTML = `<p style="color:#aaa;font-size:12px;">Analysing ${comments.length} comments…</p>`;
      const result = await analyzeVideo(videoId, comments);
      if (!result) return;

      // 4. Compute client-side metrics (need original comment data for authorId / word count)
      const totalComments = comments.length;
      const uniqueCommenters = new Set(comments.map(c => c.authorId)).size;
      const totalWords = comments.reduce(
        (sum, c) => sum + c.text.split(/\s+/).filter(w => w.length > 0).length, 0
      );
      const avgWordLength = (totalWords / totalComments).toFixed(2);
      const totalSentimentScore = result.predictions.reduce(
        (sum, p) => sum + parseInt(p.sentiment), 0
      );
      const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
      const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);

      // 5. Build the object to cache and render
      const cacheData = {
        predictions:     result.predictions,
        sentimentCounts: result.sentiment_counts,
        metrics: { totalComments, uniqueCommenters, avgWordLength, normalizedSentimentScore },
        chartImage:      result.chart_image,
        trendImage:      result.trend_image,
        wordcloudImage:  result.wordcloud_image
      };

      // 6. Store in local cache (best-effort)
      await setLocalCache(videoId, cacheData);

      // 7. Render
      renderResults(videoId, cacheData, false);

    } catch (err) {
      console.error('Popup error:', err);
      outputDiv.innerHTML = `<p style="color:#f44336;font-size:12px;">Error: ${err.message}</p>`;
    }
  });
});
