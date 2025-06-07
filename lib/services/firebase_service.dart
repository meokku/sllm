import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:swlab_sllm_app/models/chat_models.dart';

class FirebaseService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;

  // 현재 사용자 ID 가져오기
  String? get currentUserId => _auth.currentUser?.uid;

  // 사용자의 모든 채팅 가져오기
  Stream<QuerySnapshot> getUserChats() {
    if (currentUserId == null) return Stream.empty();

    return _firestore
        .collection('users')
        .doc(currentUserId)
        .collection('chats')
        .orderBy('timestamp', descending: true)
        .snapshots();
  }

  // 특정 채팅의 모든 메시지 가져오기
  Stream<QuerySnapshot> getChatMessages(String chatId) {
    if (currentUserId == null) return Stream.empty();

    return _firestore
        .collection('users')
        .doc(currentUserId)
        .collection('chats')
        .doc(chatId)
        .collection('messages')
        .orderBy('timestamp')
        .snapshots();
  }

  // 새 채팅 생성
  Future<String> createChat() async {
    if (currentUserId == null) throw Exception('로그인이 필요합니다');

    try {
      // 채팅 문서 생성
      final chatRef = _firestore
          .collection('users')
          .doc(currentUserId)
          .collection('chats')
          .doc();

      // 데이터 준비 (timestamp는 나중에 설정)
      final Map<String, dynamic> chatData = {
        'id': chatRef.id,
        'title': '새 채팅',
        'lastMessage': '',
      };

      // serverTimestamp 추가
      chatData['timestamp'] = FieldValue.serverTimestamp();

      // 문서 생성
      await chatRef.set(chatData);

      return chatRef.id;
    } catch (e) {
      print('채팅 생성 오류: $e');
      throw Exception('채팅 생성 실패: $e');
    }
  }

  // 메시지 저장
  Future<void> saveMessage(String chatId, Message message) async {
    if (currentUserId == null) throw Exception('로그인이 필요합니다');

    // 채팅 문서 참조
    final chatRef = _firestore
        .collection('users')
        .doc(currentUserId)
        .collection('chats')
        .doc(chatId);

    // 메시지 문서 생성
    await chatRef.collection('messages').add({
      'content': message.content,
      'isUser': message.isUser,
      'timestamp': Timestamp.fromDate(message.timestamp),
    });

    // 채팅 문서 업데이트 (merge 옵션 사용)
    await chatRef.set({
      'lastMessage': message.content,
      'timestamp': Timestamp.fromDate(message.timestamp),
    }, SetOptions(merge: true)); // merge 옵션을 사용하여 문서가 없어도 생성되게 함
  }

  // 채팅 제목 업데이트
  Future<void> updateChatTitle(String chatId, String newTitle) async {
    if (currentUserId == null) throw Exception('로그인이 필요합니다');

    await _firestore
        .collection('users')
        .doc(currentUserId)
        .collection('chats')
        .doc(chatId)
        .update({
      'title': newTitle,
    });
  }

  // 채팅 삭제
  Future<void> deleteChat(String chatId) async {
    if (currentUserId == null) throw Exception('로그인이 필요합니다');

    // 트랜잭션으로 채팅과 모든 메시지 삭제 (원자적 작업)
    final chatRef = _firestore
        .collection('users')
        .doc(currentUserId)
        .collection('chats')
        .doc(chatId);

    final messagesRef = chatRef.collection('messages');

    // 메시지 삭제 (일괄 삭제는 Firestore에서 제한이 있어 배치로 처리)
    final messagesSnapshot = await messagesRef.get();
    final batch = _firestore.batch();

    for (var doc in messagesSnapshot.docs) {
      batch.delete(doc.reference);
    }

    // 채팅 문서 삭제
    batch.delete(chatRef);

    // 배치 커밋
    await batch.commit();
  }
}
