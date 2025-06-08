import 'dart:convert';
import 'package:http/http.dart' as http;

class LlmService {
  final String baseUrl;
  LlmService({required this.baseUrl});

  Future<String> askLlama(String question) async {
    try {
      final response = await http
          .post(
            Uri.parse('$baseUrl/predict/'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'text': question}),
          )
          .timeout(const Duration(seconds: 120));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['generated_text'] ?? '응답에 generated_text 필드가 없습니다.';
      } else {
        return '오류 발생: ${response.statusCode} - ${response.body}';
      }
    } catch (e) {
      return '오류 발생: $e';
    }
  }

  // 스트리밍 대응용
  Stream<String> askLlamaStream(String question) async* {
    final client = http.Client();
    final request = http.Request(
      'POST',
      Uri.parse('$baseUrl/predict/'),
    );
    request.headers.addAll({'Content-Type': 'application/json'});
    request.body = jsonEncode({'text': question});

    try {
      final streamedResponse = await client.send(request);

      // stream을 utf8 decoder로 변환 후 chunk 단위로 yield
      final stream = streamedResponse.stream.transform(utf8.decoder);

      await for (final chunk in stream) {
        yield chunk;
      }
    } catch (e) {
      yield '오류 발생: $e';
    } finally {
      client.close();
    }
  }

  Future<bool> checkStatus() async {
    try {
      final response = await http
          .get(
            Uri.parse('$baseUrl/status'),
          )
          .timeout(const Duration(seconds: 5));

      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}
