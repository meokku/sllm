import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:swlab_sllm_app/providers/active_chat_provider.dart';
import 'package:swlab_sllm_app/providers/chat_session_provider.dart';
import 'package:swlab_sllm_app/screens/chat_screen.dart';
import 'package:swlab_sllm_app/services/auth_service.dart';
import 'package:swlab_sllm_app/theme/colors.dart';
import 'package:swlab_sllm_app/utils/profile_menu.dart';

class HomeScreen extends StatefulWidget {
  final String? initialUserName;

  const HomeScreen({super.key, required this.initialUserName});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  final AuthService _authService = AuthService();
  final TextEditingController _textController = TextEditingController();
  OverlayEntry? _overlayEntry;
  final GlobalKey _profileButtonKey = GlobalKey();

  final double minHeight = 600;

  // 사이드바 관련 변수
  bool _isSidebarExpanded = true;
  late AnimationController _sidebarAnimationController;
  late Animation<double> _sidebarAnimation;

  // 유저 정보 관련 변수
  bool _isLoadingUserInfo = true;
  String? _userName;

  // 오버레이 제거 함수
  void _removeOverlay() {
    _overlayEntry?.remove();
    _overlayEntry = null;
  }

  // 사이드바 토글 함수
  void _toggleSidebar() {
    setState(() {
      _isSidebarExpanded = !_isSidebarExpanded;
      if (_isSidebarExpanded) {
        _sidebarAnimationController.reverse();
      } else {
        _sidebarAnimationController.forward();
      }
    });
  }

  void _showProfileMenu() {
    // 현재 오버레이가 표시 중이면 제거
    _removeOverlay();

    // ProfileMenuUtil 사용 - 사용자 정보를 전달할 필요 없음
    ProfileMenuUtil.showProfileMenu(
      context,
      buttonKey: _profileButtonKey,
      onRemove: _removeOverlay,
      onOverlayCreated: (entry) {
        _overlayEntry = entry;
      },
    );
  }

  @override
  void initState() {
    super.initState();

    // 사이드바 애니메이션 컨트롤러 초기화
    _sidebarAnimationController = AnimationController(
      vsync: this,
      duration: Duration(milliseconds: 300),
    );

    // 사이드바 너비 애니메이션 (200 <-> 0)
    _sidebarAnimation = Tween<double>(
      begin: 200.0,
      end: 0,
    ).animate(CurvedAnimation(
      parent: _sidebarAnimationController,
      curve: Curves.easeInOut,
    ));

    // Wrapper에서 전달받은 사용자 이름이 있으면 사용
    if (widget.initialUserName != null) {
      setState(() {
        _userName = widget.initialUserName;
        _isLoadingUserInfo = false;
      });
    } else {
      // 전달받은 이름이 없는 경우에는 간소화된 방식으로 정보 로드
      _loadUserInfoSimplified();
    }
  }

  // 사용자 정보를 로드하는 메서드
  Future<void> _loadUserInfoSimplified() async {
    if (!mounted) return;

    setState(() {
      _isLoadingUserInfo = true;
    });

    try {
      final user = _authService.currentUser;
      if (mounted) {
        setState(() {
          _userName = user?.displayName;
          _isLoadingUserInfo = false;
        });
      }
    } catch (e) {
      print('사용자 정보 로드 오류: $e');
      if (mounted) {
        setState(() {
          _isLoadingUserInfo = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _removeOverlay();
    _textController.dispose();
    _sidebarAnimationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final chatSessionProvider = Provider.of<ChatSessionProvider>(context);
    final activeChatProvider = Provider.of<ActiveChatProvider>(context);
    final user = _authService.currentUser;

    // 화면 크기 가져오기
    final screenSize = MediaQuery.of(context).size;
    final screenHeight = screenSize.height;

    // 최소 높이 적용
    final containerHeight = screenHeight > minHeight ? screenHeight : minHeight;

    // 사용자 정보 상태에 따른 표시
    Widget buildGreeting() {
      if (_isLoadingUserInfo) {
        return Column(
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 10),
            Text("사용자 정보를 불러오는 중...", style: TextStyle(fontSize: 18)),
          ],
        );
      } else {
        return Text(
          "안녕하세요, ${_userName ?? '사용자'} 님",
          style: TextStyle(fontSize: 30, fontWeight: FontWeight.bold),
        );
      }
    }

    return Scaffold(
      body: Stack(
        children: [
          // 사이드바
          AnimatedBuilder(
              animation: _sidebarAnimation,
              builder: (context, child) {
                // 사이드바가 일정 너비 이하로 줄어들면 내부 콘텐츠를 숨김
                final bool showContent =
                    _sidebarAnimation.value > 120; // 임계값 설정

                return Container(
                  width: _sidebarAnimation.value,
                  color: Colors.grey[100],
                  child: Column(
                    children: [
                      // 로고 자리 공간 확보 (투명)
                      SizedBox(height: 60),
                      if (_isSidebarExpanded && showContent) ...[
                        SizedBox(
                          height: 25,
                        ),
                        // 새 채팅 버튼
                        Material(
                          color: Colors.transparent, // 배경색 투명하게 설정
                          child: InkWell(
                            onTap: () {
                              // 이미 홈 화면에 있으므로, 텍스트 필드에 포커스
                              FocusScope.of(context).requestFocus(FocusNode());
                              _textController.clear();
                            },
                            child: Padding(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 12.0, vertical: 10),
                              child: Row(
                                children: [
                                  Container(
                                    width: 24,
                                    height: 24,
                                    decoration: BoxDecoration(
                                      color: SKKUColors.green,
                                      shape: BoxShape.circle,
                                    ),
                                    child: Center(
                                      child: Icon(
                                        Icons.add,
                                        color: Colors.white,
                                        size: 16,
                                      ),
                                    ),
                                  ),
                                  SizedBox(width: 8),
                                  Text(
                                    "새 채팅",
                                    style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                        SizedBox(
                          height: 30,
                        ),
                        // 최근 항목 텍스트
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 12.0),
                          child: Align(
                            alignment: Alignment.centerLeft,
                            child: Text(
                              "최근 항목",
                              style: TextStyle(
                                color: Colors.grey[600],
                                fontSize: 14,
                              ),
                            ),
                          ),
                        ),
                        SizedBox(
                          height: 10,
                        ),
                        // 채팅 목록
                        Expanded(
                          child: chatSessionProvider.isLoading
                              ? Center(child: CircularProgressIndicator())
                              : ListView.builder(
                                  itemCount:
                                      chatSessionProvider.chatSessions.length,
                                  itemBuilder: (context, index) {
                                    final chat =
                                        chatSessionProvider.chatSessions[index];

                                    return Material(
                                      color: Colors.grey[100],
                                      child: InkWell(
                                        onTap: () {
                                          // 해당 채팅만 선택 (새 채팅 생성 안 함)
                                          chatSessionProvider
                                              .selectChat(chat.id);

                                          // 채팅 화면으로 이동
                                          Navigator.pushReplacementNamed(
                                              context, '/chat');
                                        },
                                        child: Container(
                                          width: double.infinity,
                                          padding: EdgeInsets.symmetric(
                                              vertical: 8, horizontal: 12),
                                          child: Text(
                                            chat.title,
                                            style: TextStyle(fontSize: 14),
                                          ),
                                        ),
                                      ),
                                    );
                                  },
                                ),
                        ),
                      ]
                    ],
                  ),
                );
              }),

          // 메인 컨텐츠 영역
          AnimatedContainer(
            duration: Duration(milliseconds: 300),
            curve: Curves.easeInOut,
            margin: EdgeInsets.only(
                left: _isSidebarExpanded ? 200 : 0), // 왼쪽 마진으로 위치 조정
            child: SizedBox(
              // 최소 높이 설정
              height: containerHeight,
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    // 상단 헤더 영역 (로고와 프로필 버튼)
                    Container(
                      height:
                          kToolbarHeight + MediaQuery.of(context).padding.top,
                      padding: EdgeInsets.only(
                          top: MediaQuery.of(context).padding.top),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.end,
                        children: [
                          Padding(
                            padding: const EdgeInsets.only(right: 15.0),
                            child: Tooltip(
                              message: "사용자 프로필",
                              child: Material(
                                key: _profileButtonKey,
                                color: Colors.transparent,
                                child: Ink(
                                  decoration: BoxDecoration(
                                    color: Colors.grey[400],
                                    shape: BoxShape.circle,
                                  ),
                                  child: InkWell(
                                    customBorder: CircleBorder(),
                                    onTap: _showProfileMenu,
                                    child: SizedBox(
                                      width: 40,
                                      height: 40,
                                      child: Icon(
                                        Icons.person,
                                        color: Colors.white,
                                        size: 30,
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),

                    // 본문 영역
                    SizedBox(
                      // 전체 높이에서 헤더 높이를 뺀 값
                      height: containerHeight -
                          (kToolbarHeight + MediaQuery.of(context).padding.top),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          buildGreeting(),
                          SizedBox(height: 40),
                          // 채팅 입력 영역
                          Center(
                            child: Container(
                              width: 1000,
                              padding: EdgeInsets.symmetric(horizontal: 20),
                              child: Container(
                                padding: EdgeInsets.all(16.0),
                                child: Row(
                                  children: [
                                    Expanded(
                                      child: Container(
                                        decoration: BoxDecoration(
                                          color: Colors.white,
                                          borderRadius:
                                              BorderRadius.circular(24.0),
                                          border: Border.all(
                                              color: Colors.grey[300]!),
                                          boxShadow: [
                                            BoxShadow(
                                              color: Colors.black
                                                  .withOpacity(0.05),
                                              blurRadius: 2,
                                              spreadRadius: 1,
                                            ),
                                          ],
                                        ),
                                        child: Column(
                                          children: [
                                            Row(
                                              children: [
                                                Expanded(
                                                  child: TextField(
                                                    maxLines: null,
                                                    textInputAction:
                                                        TextInputAction.send,
                                                    keyboardType:
                                                        TextInputType.text,
                                                    controller: _textController,
                                                    decoration: InputDecoration(
                                                      hintText: '무엇을 도와드릴까요?',
                                                      hintStyle: TextStyle(
                                                        color: Colors.grey,
                                                      ),
                                                      border: InputBorder.none,
                                                      contentPadding:
                                                          EdgeInsets.symmetric(
                                                              vertical: 20,
                                                              horizontal: 20.0),
                                                    ),
                                                    onSubmitted: (text) async {
                                                      if (!activeChatProvider
                                                              .isLoading &&
                                                          _textController.text
                                                              .isNotEmpty) {
                                                        try {
                                                          // 1. 새 채팅 생성 (비동기 메서드)
                                                          await chatSessionProvider
                                                              .createNewChat();

                                                          // 2. 초기 메시지 설정 (로딩 상태가 됨)
                                                          activeChatProvider
                                                              .setInitialMessage(
                                                                  _textController
                                                                      .text);

                                                          // 3. 텍스트 컨트롤러 초기화
                                                          _textController
                                                              .clear();

                                                          // 4. 채팅 화면으로 즉시 이동
                                                          Navigator
                                                              .pushReplacementNamed(
                                                                  context,
                                                                  '/chat');

                                                          // 5. 실제 메시지 전송은 background에서 진행
                                                          activeChatProvider
                                                              .sendMessage(
                                                                  text);
                                                        } catch (e) {
                                                          // 오류 처리
                                                          ScaffoldMessenger.of(
                                                                  context)
                                                              .showSnackBar(
                                                            SnackBar(
                                                                content: Text(
                                                                    '오류가 발생했습니다: $e')),
                                                          );
                                                        }
                                                      }
                                                    },
                                                  ),
                                                ),
                                                Container(
                                                  margin: EdgeInsets.only(
                                                      right: 10, top: 10),
                                                  decoration: BoxDecoration(
                                                    color: SKKUColors.green,
                                                    shape: BoxShape.circle,
                                                  ),
                                                  child: IconButton(
                                                    icon: Icon(
                                                        Icons.arrow_upward,
                                                        color: Colors.white),
                                                    onPressed: () async {
                                                      if (!activeChatProvider
                                                              .isLoading &&
                                                          _textController.text
                                                              .isNotEmpty) {
                                                        try {
                                                          // 1. 새 채팅 생성
                                                          await chatSessionProvider
                                                              .createNewChat();

                                                          // 2. 초기 메시지 설정 (로딩 상태가 됨)
                                                          activeChatProvider
                                                              .setInitialMessage(
                                                                  _textController
                                                                      .text);

                                                          // 3. 임시 저장된 텍스트 변수
                                                          final text =
                                                              _textController
                                                                  .text;

                                                          // 4. 텍스트 컨트롤러 초기화
                                                          _textController
                                                              .clear();

                                                          // 5. 채팅 화면으로 즉시 이동
                                                          Navigator
                                                              .pushReplacementNamed(
                                                                  context,
                                                                  '/chat');

                                                          // 6. 실제 메시지 전송은 background에서 진행
                                                          activeChatProvider
                                                              .sendMessage(
                                                                  text);
                                                        } catch (e) {
                                                          // 오류 처리
                                                          ScaffoldMessenger.of(
                                                                  context)
                                                              .showSnackBar(
                                                            SnackBar(
                                                                content: Text(
                                                                    '오류가 발생했습니다: $e')),
                                                          );
                                                        }
                                                      }
                                                    },
                                                  ),
                                                ),
                                              ],
                                            ),
                                            // 추가 입력 옵션 (+ 버튼)
                                            Container(
                                              alignment: Alignment.centerLeft,
                                              padding: EdgeInsets.only(
                                                  left: 10.0, bottom: 10.0),
                                              child: Container(
                                                width: 32,
                                                height: 32,
                                                decoration: BoxDecoration(
                                                  shape: BoxShape.circle,
                                                  border: Border.all(
                                                      color: Colors.grey[300]!),
                                                ),
                                                child: IconButton(
                                                  icon: Icon(Icons.add,
                                                      size: 20,
                                                      color: Colors.grey),
                                                  onPressed: () {
                                                    // 추가 버튼 클릭 시
                                                    // 파일 추가 등의 로직 설정
                                                  },
                                                  padding: EdgeInsets.zero,
                                                  constraints: BoxConstraints(),
                                                ),
                                              ),
                                            )
                                          ],
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // 로고 및 토글 버튼 (위 레이어)
          Positioned(
            top: 0,
            left: 0,
            height: 60,
            width: 200,
            child: SizedBox(
              child: Padding(
                padding: const EdgeInsets.only(left: 15.0, top: 13, right: 5),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Image.asset(
                      "assets/images/SKKULLM.png",
                      width: 120,
                    ),
                    IconButton(
                      onPressed: _toggleSidebar,
                      icon: Icon(_isSidebarExpanded
                          ? Icons.keyboard_double_arrow_left_rounded
                          : Icons.keyboard_double_arrow_right_rounded),
                      tooltip: _isSidebarExpanded ? "사이드바 접기" : "사이드바 펼치기",
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
