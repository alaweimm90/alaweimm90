package com.atlas.intellij.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.components.Service;
import com.intellij.openapi.diagnostic.Logger;
import org.apache.http.client.methods.*;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.CompletableFuture;

@Service
public final class AtlasService {
    private static final Logger LOG = Logger.getInstance(AtlasService.class);
    private static final String DEFAULT_API_URL = "https://api.atlas-platform.com/v1";

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final CloseableHttpClient httpClient = HttpClients.createDefault();

    private String apiKey;
    private String apiUrl = DEFAULT_API_URL;

    public static AtlasService getInstance() {
        return ApplicationManager.getApplication().getService(AtlasService.class);
    }

    public void setApiKey(@Nullable String apiKey) {
        this.apiKey = apiKey;
    }

    public void setApiUrl(@NotNull String apiUrl) {
        this.apiUrl = apiUrl;
    }

    public boolean isConfigured() {
        return apiKey != null && !apiKey.trim().isEmpty();
    }

    // Task Management
    public CompletableFuture<AtlasTask> submitTask(@NotNull String type, @NotNull String description) {
        return submitTask(type, description, null, null);
    }

    public CompletableFuture<AtlasTask> submitTask(@NotNull String type, @NotNull String description,
                                                  @Nullable Map<String, Object> context,
                                                  @Nullable Map<String, Object> requirements) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                Map<String, Object> requestBody = new HashMap<>();
                requestBody.put("type", type);
                requestBody.put("description", description);
                if (context != null) requestBody.put("context", context);
                if (requirements != null) requestBody.put("requirements", requirements);

                String jsonBody = objectMapper.writeValueAsString(requestBody);
                HttpPost request = createPostRequest("/tasks", jsonBody);

                try (CloseableHttpResponse response = httpClient.execute(request)) {
                    String responseBody = EntityUtils.toString(response.getEntity());
                    Map<String, Object> responseData = objectMapper.readValue(responseBody,
                        new TypeReference<Map<String, Object>>() {});

                    if ((Boolean) responseData.get("success")) {
                        Map<String, Object> data = (Map<String, Object>) responseData.get("data");
                        return new AtlasTask((String) data.get("task_id"), (String) data.get("status"));
                    } else {
                        throw new RuntimeException("Task submission failed: " + responseData.get("error"));
                    }
                }
            } catch (Exception e) {
                LOG.error("Failed to submit task", e);
                throw new RuntimeException("Failed to submit task: " + e.getMessage(), e);
            }
        });
    }

    public CompletableFuture<AtlasTask> getTask(@NotNull String taskId) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                HttpGet request = createGetRequest("/tasks/" + taskId);

                try (CloseableHttpResponse response = httpClient.execute(request)) {
                    String responseBody = EntityUtils.toString(response.getEntity());
                    Map<String, Object> responseData = objectMapper.readValue(responseBody,
                        new TypeReference<Map<String, Object>>() {});

                    if ((Boolean) responseData.get("success")) {
                        Map<String, Object> data = (Map<String, Object>) responseData.get("data");
                        return parseTaskFromResponse(data);
                    } else {
                        throw new RuntimeException("Failed to get task: " + responseData.get("error"));
                    }
                }
            } catch (Exception e) {
                LOG.error("Failed to get task", e);
                throw new RuntimeException("Failed to get task: " + e.getMessage(), e);
            }
        });
    }

    // Agent Management
    public CompletableFuture<List<AtlasAgent>> listAgents() {
        return CompletableFuture.supplyAsync(() -> {
            try {
                HttpGet request = createGetRequest("/agents");

                try (CloseableHttpResponse response = httpClient.execute(request)) {
                    String responseBody = EntityUtils.toString(response.getEntity());
                    Map<String, Object> responseData = objectMapper.readValue(responseBody,
                        new TypeReference<Map<String, Object>>() {});

                    if ((Boolean) responseData.get("success")) {
                        Map<String, Object> data = (Map<String, Object>) responseData.get("data");
                        List<Map<String, Object>> agentsData = (List<Map<String, Object>>) data.get("agents");
                        return agentsData.stream().map(this::parseAgentFromResponse).toList();
                    } else {
                        throw new RuntimeException("Failed to list agents: " + responseData.get("error"));
                    }
                }
            } catch (Exception e) {
                LOG.error("Failed to list agents", e);
                throw new RuntimeException("Failed to list agents: " + e.getMessage(), e);
            }
        });
    }

    // Analysis
    public CompletableFuture<AtlasAnalysis> startAnalysis(@NotNull Map<String, Object> options) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                String jsonBody = objectMapper.writeValueAsString(options);
                HttpPost request = createPostRequest("/analyze", jsonBody);

                try (CloseableHttpResponse response = httpClient.execute(request)) {
                    String responseBody = EntityUtils.toString(response.getEntity());
                    Map<String, Object> responseData = objectMapper.readValue(responseBody,
                        new TypeReference<Map<String, Object>>() {});

                    if ((Boolean) responseData.get("success")) {
                        Map<String, Object> data = (Map<String, Object>) responseData.get("data");
                        return new AtlasAnalysis((String) data.get("analysis_id"), (String) data.get("status"));
                    } else {
                        throw new RuntimeException("Analysis failed: " + responseData.get("error"));
                    }
                }
            } catch (Exception e) {
                LOG.error("Failed to start analysis", e);
                throw new RuntimeException("Failed to start analysis: " + e.getMessage(), e);
            }
        });
    }

    // Helper methods
    private HttpGet createGetRequest(String endpoint) {
        HttpGet request = new HttpGet(apiUrl + endpoint);
        request.setHeader("Authorization", "Bearer " + apiKey);
        request.setHeader("Content-Type", "application/json");
        return request;
    }

    private HttpPost createPostRequest(String endpoint, String jsonBody) {
        HttpPost request = new HttpPost(apiUrl + endpoint);
        request.setHeader("Authorization", "Bearer " + apiKey);
        request.setHeader("Content-Type", "application/json");
        request.setEntity(new StringEntity(jsonBody));
        return request;
    }

    private AtlasTask parseTaskFromResponse(Map<String, Object> data) {
        return new AtlasTask(
            (String) data.get("task_id"),
            (String) data.get("type"),
            (String) data.get("status"),
            (String) data.get("description"),
            data.get("result"),
            (String) data.get("agent_id"),
            (String) data.get("created_at"),
            (String) data.get("completed_at"),
            data.get("cost_usd") != null ? ((Number) data.get("cost_usd")).doubleValue() : null,
            (String) data.get("error")
        );
    }

    private AtlasAgent parseAgentFromResponse(Map<String, Object> data) {
        return new AtlasAgent(
            (String) data.get("agent_id"),
            (String) data.get("name"),
            (String) data.get("provider"),
            (String) data.get("model"),
            (String) data.get("status"),
            (List<String>) data.get("capabilities")
        );
    }

    // Data classes
    public static class AtlasTask {
        public final String taskId;
        public final String type;
        public final String status;
        public final String description;
        public final Object result;
        public final String agentId;
        public final String createdAt;
        public final String completedAt;
        public final Double costUsd;
        public final String error;

        public AtlasTask(String taskId, String type, String status, String description,
                        Object result, String agentId, String createdAt, String completedAt,
                        Double costUsd, String error) {
            this.taskId = taskId;
            this.type = type;
            this.status = status;
            this.description = description;
            this.result = result;
            this.agentId = agentId;
            this.createdAt = createdAt;
            this.completedAt = completedAt;
            this.costUsd = costUsd;
            this.error = error;
        }

        public AtlasTask(String taskId, String status) {
            this(taskId, null, status, null, null, null, null, null, null, null);
        }
    }

    public static class AtlasAgent {
        public final String agentId;
        public final String name;
        public final String provider;
        public final String model;
        public final String status;
        public final List<String> capabilities;

        public AtlasAgent(String agentId, String name, String provider, String model,
                         String status, List<String> capabilities) {
            this.agentId = agentId;
            this.name = name;
            this.provider = provider;
            this.model = model;
            this.status = status;
            this.capabilities = capabilities;
        }
    }

    public static class AtlasAnalysis {
        public final String analysisId;
        public final String status;

        public AtlasAnalysis(String analysisId, String status) {
            this.analysisId = analysisId;
            this.status = status;
        }
    }
}