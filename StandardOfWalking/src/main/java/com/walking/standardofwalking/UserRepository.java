package com.walking.standardofwalking;

import org.springframework.data.jpa.repository.JpaRepository;

//class 타입과 pk타입 설정하면 엔티티와 매핑되는 테이블의 CRUD 사용할 수 있음
public interface UserRepository extends JpaRepository<User, Long> {

}
