// Exey Panteleev
import Foundation

// MARK: - AI Models

struct AIInsight: Codable {
    let file: String
    let purpose: String
    let complexity: Int
    let keyComponents: [String]
    let refactoringTips: [String]
    let securityConcerns: [String]
    let businessImpact: String
    let readingTime: Int
}

struct AIConversationResponse: Codable {
    let answer: String
    let suggestedFiles: [String]
    let confidence: Double
}

struct AnalysisContext {
    let totalFiles: Int
    let dependencies: Int
    let dependents: Int
    let pageRank: Double
    let gitChurn: Int
}

struct CodebaseContext {
    let totalFiles: Int
    let languages: [String]
    let hotspots: [String]
    var files: [FileDigest] = []
}

struct FileDigest {
    let path: String
    let absolutePath: String
    let types: [String]
    let lineCount: Int
}

// MARK: - AI Code Analyzer

/// AI-powered code analysis using URLSession (Apple-native networking).
/// Supports Anthropic Claude, Google Gemini, and GitHub Copilot Enterprise.
final class AICodeAnalyzer: Sendable {
    let apiKey: String
    let model: String
    let provider: String
    let baseURL: String
    let tokenURL: String
    private let copilotToken = CopilotTokenProvider()

    var isConfigured: Bool {
        !apiKey.isEmpty && apiKey != "heuristic" && !apiKey.hasPrefix("demo")
    }

    init(
        apiKey: String,
        model: String = "claude-sonnet-4-20250514",
        provider: String = "anthropic",
        baseURL: String = "",
        tokenURL: String = ""
    ) {
        self.apiKey = apiKey
        self.model = model
        self.provider = provider
        self.baseURL = baseURL
        self.tokenURL = tokenURL
    }

    // MARK: - Public API

    func analyzeFile(_ file: ParsedFile, context: AnalysisContext) async throws -> AIInsight {
        guard isConfigured else {
            throw CodeContextError.aiProvider("AI not configured. Set API key in .codecontext.json")
        }

        let prompt = buildFileAnalysisPrompt(file: file, context: context)
        let response = try await callAI(prompt: prompt)
        return parseInsight(response: response, filePath: file.filePath)
    }

    func batchAnalyze(
        files: [ParsedFile],
        graph: DependencyGraph,
        limit: Int = 50
    ) async -> [String: AIInsight] {
        let prioritized = files
            .sorted { (graph.pageRankScores[$0.filePath] ?? 0) > (graph.pageRankScores[$1.filePath] ?? 0) }
            .prefix(limit)

        print("🤖 AI analyzing top \(limit) files...")

        var results: [String: AIInsight] = [:]

        await withTaskGroup(of: (String, AIInsight)?.self) { group in
            for file in prioritized {
                group.addTask {
                    let context = AnalysisContext(
                        totalFiles: files.count,
                        dependencies: graph.outDegree(of: file.filePath),
                        dependents: graph.inDegree(of: file.filePath),
                        pageRank: graph.pageRankScores[file.filePath] ?? 0,
                        gitChurn: file.gitMetadata.changeFrequency
                    )
                    do {
                        let insight = try await self.analyzeFile(file, context: context)
                        return (file.filePath, insight)
                    } catch {
                        print("⚠️  AI analysis failed for \(file.fileName): \(error.localizedDescription)")
                        return nil
                    }
                }
            }

            for await result in group {
                if let (path, insight) = result {
                    results[path] = insight
                }
            }
        }

        return results
    }

    func askQuestion(_ question: String, context: CodebaseContext) async throws -> AIConversationResponse {
        let prompt = buildConversationPrompt(question: question, context: context)
        let response = try await callAI(prompt: prompt)
        return parseConversation(response: response)
    }

    // MARK: - Deep (agentic) Q&A

    func askQuestionDeep(
        _ question: String,
        context: CodebaseContext,
        onProgress: @Sendable (String) -> Void = { _ in }
    ) async throws -> AIConversationResponse {
        onProgress("selecting relevant files…")
        let selected = try await selectFiles(question: question, context: context)
        guard !selected.isEmpty else {
            onProgress("no selection — falling back to one-shot")
            return try await askQuestion(question, context: context)
        }
        onProgress("reading \(selected.count) files: \(selected.map(\.path).joined(separator: ", "))")

        let evidence = buildEvidence(files: selected, question: question)
        let prompt = buildDeepAnswerPrompt(question: question, evidence: evidence)
        let response = try await callAI(prompt: prompt)
        return parseConversation(response: response)
    }

    private func selectFiles(question: String, context: CodebaseContext) async throws -> [FileDigest] {
        let prompt = """
        You're locating code in a Swift codebase. From the repo map, pick the files
        whose contents are most likely needed to answer the question. Choose up to 8,
        most relevant first. Copy paths exactly as written; do not invent paths.

        REPO MAP (path — declared types, ranked by importance):
        \(renderRepoMap(context, budget: 14000))
        QUESTION: "\(question)"

        Respond ONLY with JSON: {"files":["path/one.swift","path/two.swift"]}
        """
        let response = try await callAI(prompt: prompt)
        let requested = parseFileList(response)

        let byPath = Dictionary(context.files.map { ($0.path, $0) }, uniquingKeysWith: { a, _ in a })
        var matched: [FileDigest] = []
        var seen = Set<String>()
        for req in requested {
            let hit = byPath[req] ?? context.files.first { $0.path.hasSuffix("/" + req) || $0.path.hasSuffix(req) }
            if let hit, seen.insert(hit.path).inserted { matched.append(hit) }
        }
        return Array(matched.prefix(12))
    }

    private func parseFileList(_ response: String) -> [String] {
        guard let json = extractJSON(from: response),
              let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let files = obj["files"] as? [String] else { return [] }
        return files
    }

    // MARK: - AI Calls (URLSession — Apple native)

    private func callAI(prompt: String) async throws -> String {
        switch provider.lowercased() {
        case "anthropic", "claude":
            return try await callClaude(prompt: prompt)
        case "gemini":
            return try await callGemini(prompt: prompt)
        case "copilot", "github", "github-copilot":
            return try await callCopilot(prompt: prompt)
        default:
            throw CodeContextError.aiProvider("Unsupported provider: \(provider)")
        }
    }

    private func callClaude(prompt: String) async throws -> String {
        var request = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")

        let body: [String: Any] = [
            "model": model,
            "max_tokens": 1000,
            "messages": [["role": "user", "content": prompt]]
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            throw CodeContextError.aiProvider("Claude API error: \(body)")
        }

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let content = (json?["content"] as? [[String: Any]])?.first
        guard let text = content?["text"] as? String else {
            throw CodeContextError.aiProvider("Invalid Claude response format")
        }
        return text
    }

    private func callGemini(prompt: String) async throws -> String {
        let urlString = "https://generativelanguage.googleapis.com/v1beta/models/\(model):generateContent?key=\(apiKey)"
        var request = URLRequest(url: URL(string: urlString)!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "contents": [["parts": [["text": prompt]]]],
            "generationConfig": [
                "temperature": 0.7,
                "maxOutputTokens": 2048
            ]
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            throw CodeContextError.aiProvider("Gemini API error: \(body)")
        }

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let candidates = json?["candidates"] as? [[String: Any]]
        let content = candidates?.first?["content"] as? [String: Any]
        let parts = content?["parts"] as? [[String: Any]]
        guard let text = parts?.first?["text"] as? String else {
            throw CodeContextError.aiProvider("Invalid Gemini response format")
        }
        return text
    }

    // MARK: - GitHub Copilot (OpenAI-compatible chat completions)

    /// Live model catalog from the Copilot `/models` endpoint.
    func listCopilotModels() async throws -> [String] {
        let request = try await copilotRequest(path: "/models", method: "GET")
        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            throw CodeContextError.aiProvider("Copilot models error: \(body)")
        }

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let models = json?["data"] as? [[String: Any]] else {
            throw CodeContextError.aiProvider("Invalid Copilot models response")
        }
        return models.compactMap { $0["id"] as? String }.sorted()
    }

    private func callCopilot(prompt: String) async throws -> String {
        var request = try await copilotRequest(path: "/chat/completions", method: "POST")
        let body: [String: Any] = [
            "model": model,
            "max_tokens": 1000,
            "messages": [["role": "user", "content": prompt]]
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            throw CodeContextError.aiProvider("Copilot API error: \(body)")
        }

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let choices = json?["choices"] as? [[String: Any]]
        let message = choices?.first?["message"] as? [String: Any]
        guard let text = message?["content"] as? String else {
            throw CodeContextError.aiProvider("Invalid Copilot response format")
        }
        return text
    }

    private func copilotRequest(path: String, method: String) async throws -> URLRequest {
        let bearer = try await copilotToken.bearer(githubToken: apiKey, tokenURL: tokenURL)
        let host = baseURL.isEmpty ? "https://api.githubcopilot.com" : baseURL.trimmingTrailingSlash
        guard let url = URL(string: host + path) else {
            throw CodeContextError.aiProvider("Invalid Copilot baseURL: \(host)")
        }

        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        request.setValue("ArchSwiftScope/1.0", forHTTPHeaderField: "Editor-Version")
        request.setValue("ArchSwiftScope/1.0", forHTTPHeaderField: "Editor-Plugin-Version")
        request.setValue("vscode-chat", forHTTPHeaderField: "Copilot-Integration-Id")
        return request
    }

    // MARK: - Prompt Builders

    private func buildFileAnalysisPrompt(file: ParsedFile, context: AnalysisContext) -> String {
        let fileContent = (try? String(contentsOfFile: file.filePath, encoding: .utf8).prefix(3000)) ?? "[unavailable]"
        return """
        Analyze this Swift codebase file and provide structured insights.

        FILE: \(file.fileName)
        MODULE: \(file.moduleName)
        IMPORTS: \(file.imports.prefix(10).joined(separator: ", "))

        CONTEXT:
        - Total codebase size: \(context.totalFiles) files
        - This file depends on: \(context.dependencies) files
        - This file is used by: \(context.dependents) files
        - PageRank (importance): \(String(format: "%.4f", context.pageRank))
        - Git churn: \(context.gitChurn) changes

        CODE PREVIEW:
        \(fileContent)

        Respond ONLY with JSON:
        {"purpose":"...","complexity":1-10,"keyComponents":[],"refactoringTips":[],"securityConcerns":[],"businessImpact":"...","readingTime":N}
        """
    }

    private func renderRepoMap(_ context: CodebaseContext, budget: Int) -> String {
        var map = ""
        var shown = 0
        for file in context.files {
            let types = file.types.prefix(8).joined(separator: ", ")
            let line = types.isEmpty
                ? "\(file.path) (\(file.lineCount) loc)\n"
                : "\(file.path) — \(types) (\(file.lineCount) loc)\n"
            if map.count + line.count > budget { break }
            map += line
            shown += 1
        }
        if shown < context.files.count {
            map += "…and \(context.files.count - shown) more files (omitted, ranked lower).\n"
        }
        return map
    }

    // MARK: - Evidence Extraction

    private static let stopWords: Set<String> = [
        "the", "and", "for", "what", "where", "which", "how", "does", "this", "that",
        "with", "from", "into", "are", "is", "was", "were", "has", "have", "can",
        "would", "should", "could", "will", "about", "code", "file", "files", "swift",
    ]

    private func keywords(from question: String) -> [String] {
        let tokens = question.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 3 && !Self.stopWords.contains($0) }
        return Array(Set(tokens))
    }

    private func buildEvidence(files: [FileDigest], question: String) -> String {
        let keys = keywords(from: question)
        let signaturePrefixes = [
            "func ", "class ", "struct ", "enum ", "protocol ", "actor ",
            "extension ", "init", "case ", "@",
        ]
        let globalBudget = 16000
        let perFileLineCap = 70
        var out = ""

        for file in files {
            guard out.count < globalBudget,
                  let content = try? String(contentsOfFile: file.absolutePath, encoding: .utf8)
            else { continue }
            let lines = content.components(separatedBy: "\n")

            var keep = Set<Int>()
            for (i, line) in lines.enumerated() {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                let isSignature = signaturePrefixes.contains { trimmed.hasPrefix($0) }
                let matchesKeyword = keys.contains { line.range(of: $0, options: .caseInsensitive) != nil }
                if isSignature || matchesKeyword {
                    if matchesKeyword {
                        for j in max(0, i - 3)...min(lines.count - 1, i + 3) { keep.insert(j) }
                    } else {
                        keep.insert(i)
                    }
                }
            }

            guard !keep.isEmpty else { continue }
            out += "\n// ===== \(file.path) =====\n"
            var lastShown = -2
            var emitted = 0
            for idx in keep.sorted() {
                if emitted >= perFileLineCap || out.count > globalBudget { out += "…\n"; break }
                if idx > lastShown + 1 { out += "…\n" }
                out += lines[idx] + "\n"
                lastShown = idx
                emitted += 1
            }
        }
        return out.isEmpty ? "[no matching lines extracted]" : out
    }

    private func buildDeepAnswerPrompt(question: String, evidence: String) -> String {
        return """
        You're an expert guide for this Swift codebase. Below are excerpts (signatures
        and question-relevant lines) from the most relevant files. Answer from this
        evidence; cite the file paths shown in the // ===== headers. Say so if the
        evidence is insufficient rather than guessing.

        EVIDENCE:
        \(evidence)

        QUESTION: "\(question)"

        Respond with JSON:
        {"answer":"...","suggestedFiles":[],"confidence":0.0-1.0}
        """
    }

    private func buildConversationPrompt(question: String, context: CodebaseContext) -> String {
        return """
        You're an expert guide for this Swift codebase. Use the repo map below to
        locate code. Cite concrete file paths from the map; don't invent paths.

        CODEBASE: \(context.totalFiles) files, Languages: \(context.languages.joined(separator: ", "))

        REPO MAP (path — declared types, ranked by importance):
        \(renderRepoMap(context, budget: 12000))
        QUESTION: "\(question)"

        Respond with JSON:
        {"answer":"...","suggestedFiles":[],"confidence":0.0-1.0}
        """
    }

    // MARK: - Response Parsers

    private func parseInsight(response: String, filePath: String) -> AIInsight {
        guard let jsonStr = extractJSON(from: response),
              let data = jsonStr.data(using: .utf8),
              let insight = try? JSONDecoder().decode(AIInsight.self, from: data) else {
            return AIInsight(
                file: filePath, purpose: "Analysis unavailable", complexity: 5,
                keyComponents: [], refactoringTips: [], securityConcerns: [],
                businessImpact: "Unknown", readingTime: 10
            )
        }
        // Override file path
        return AIInsight(
            file: filePath, purpose: insight.purpose, complexity: insight.complexity,
            keyComponents: insight.keyComponents, refactoringTips: insight.refactoringTips,
            securityConcerns: insight.securityConcerns, businessImpact: insight.businessImpact,
            readingTime: insight.readingTime
        )
    }

    private func parseConversation(response: String) -> AIConversationResponse {
        guard let jsonStr = extractJSON(from: response),
              let data = jsonStr.data(using: .utf8),
              let result = try? JSONDecoder().decode(AIConversationResponse.self, from: data) else {
            return AIConversationResponse(
                answer: String(response.prefix(200)) + "...",
                suggestedFiles: [], confidence: 0.5
            )
        }
        return result
    }

    private func extractJSON(from text: String) -> String? {
        guard let start = text.firstIndex(of: "{"),
              let end = text.lastIndex(of: "}") else { return nil }
        return String(text[start...end])
    }
}

// MARK: - Copilot Token Provider

/// Exchanges a GitHub token for a short-lived Copilot session token, cached across calls.
private actor CopilotTokenProvider {
    private var cached: (token: String, expiresAt: Date)?

    func bearer(githubToken: String, tokenURL: String) async throws -> String {
        if githubToken.contains("tid=") { return githubToken }

        if let cached, cached.expiresAt > Date().addingTimeInterval(60) {
            return cached.token
        }

        let endpoint = tokenURL.isEmpty
            ? "https://api.github.com/copilot_internal/v2/token"
            : tokenURL
        guard let url = URL(string: endpoint) else {
            throw CodeContextError.aiProvider("Invalid Copilot tokenURL: \(endpoint)")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("token \(githubToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("ArchSwiftScope/1.0", forHTTPHeaderField: "Editor-Version")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            throw CodeContextError.aiProvider(
                "Copilot token exchange failed (ai.apiKey must be a GitHub token with Copilot access): \(body)"
            )
        }

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let token = json?["token"] as? String else {
            throw CodeContextError.aiProvider("Invalid Copilot token response")
        }
        let expiry = (json?["expires_at"] as? TimeInterval).map(Date.init(timeIntervalSince1970:))
            ?? Date().addingTimeInterval(300)
        cached = (token, expiry)
        return token
    }
}

private extension String {
    var trimmingTrailingSlash: String {
        hasSuffix("/") ? String(dropLast()) : self
    }
}
